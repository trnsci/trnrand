#!/usr/bin/env bash
#
# Run neuron-marked pytest tests on the trnrand CI instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_tests.sh [instance_type]
#
#   For trn2 (sa-east-1, provisioned via infra/terraform-trn2/):
#   AWS_PROFILE=aws AWS_REGION=sa-east-1 ./scripts/run_neuron_tests.sh trn2
#
# Default instance_type is trn1 (looks for Name=trnrand-ci-trn1).
# Provision with:
#   trn1: cd infra/terraform    && terraform apply -var=vpc_id=... -var=subnet_id=...
#   trn2: cd infra/terraform-trn2 && terraform apply -var=vpc_id=... -var=subnet_id=...
#
# This script:
#   1. Starts the tagged instance (if stopped)
#   2. Waits for SSM agent
#   3. Runs `pytest tests/ -v -m neuron` via SSM send-command
#   4. Prints stdout/stderr
#   5. Stops the instance (always, even on failure)

set -euo pipefail

WARM=0
PHILOX_ONLY=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --warm) WARM=1 ;;
    --philox-only) PHILOX_ONLY=1 ;;
    *) echo "ERROR: unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

INSTANCE_TYPE="${1:-trn1}"
TAG="trnrand-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_neuron_tests.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
  echo "Provision with: cd infra/terraform && terraform apply" >&2
  exit 1
fi

echo "Instance: $INSTANCE_ID"

cleanup() {
  local exit_code=$?
  echo ""
  echo "Stopping $INSTANCE_ID..."
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  exit "$exit_code"
}
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)

if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi

echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
# `aws ssm wait instance-information` isn't available in all CLI versions —
# poll describe-instance-information instead.
for _ in $(seq 1 60); do
  PING=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$PING" == "Online" ]] && break
  sleep 5
done
if [[ "$PING" != "Online" ]]; then
  echo "ERROR: SSM agent not Online after 5 minutes (last PingStatus=$PING)" >&2
  exit 1
fi

# --philox-only: deselect the two Box-Muller kernel tests so we can see
# Philox's hardware status without Box-Muller's separate trn1 compile
# issue (NCC_IBIR605) masking it. Simulator has already validated
# Box-Muller; isolating Philox here gives independent hardware signal.
# Uses explicit --deselect node-ids (not -k) to avoid nested-quote issues
# in the SSM command parameter.
if [[ "$PHILOX_ONLY" == "1" ]]; then
  PYTEST_DESELECT="--deselect tests/test_nki_philox.py::TestPhiloxNKI::test_box_muller_kernel_matches_reference --deselect tests/test_nki_philox.py::TestPhiloxNKI::test_box_muller_kernel_distribution"
else
  PYTEST_DESELECT=""
fi

# --warm: run the suite twice to expose the NEFF cache delta — the second
# pass gets warm /var/tmp/neuron-compile-cache/. -s surfaces the perf
# prints from TestPerformance.
if [[ "$WARM" == "1" ]]; then
  PYTEST_INVOCATION="\$NEURON_VENV/bin/pytest /home/ubuntu/trnrand/tests/ -v -s -m neuron $PYTEST_DESELECT --tb=short && echo === WARM PASS === && \$NEURON_VENV/bin/pytest /home/ubuntu/trnrand/tests/ -v -s -m neuron $PYTEST_DESELECT --tb=short"
else
  PYTEST_INVOCATION="\$NEURON_VENV/bin/pytest /home/ubuntu/trnrand/tests/ -v -m neuron $PYTEST_DESELECT --tb=short"
fi

echo "Sending test command (SHA=$SHA, warm=$WARM)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnrand neuron tests @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trnrand && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && NEURON_VENV=\$(ls -d /opt/aws_neuronx_venv_pytorch_* | head -1) && sudo -u ubuntu \$NEURON_VENV/bin/pip install -e /home/ubuntu/trnrand[dev] --quiet && sudo -u ubuntu env PATH=\$NEURON_VENV/bin:/usr/bin:/bin TMPDIR=/var/tmp TRNRAND_REQUIRE_NKI=1 NEURON_CC_FLAGS=\\\"--optlevel=1 --retry_failed_compilation\\\" $PYTEST_INVOCATION'\"
  ]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (this may take several minutes — cold NKI compile + first-run pip install can run long)..."

# The AWS CLI `ssm wait command-executed` waiter maxes out at ~100 attempts
# × 5s = 8 min. Cold NKI compile for a fresh repo + pip install into the
# Neuron venv can exceed that on first run. Poll manually with a longer
# ceiling (30 min).
POLL_TIMEOUT=1800
POLL_INTERVAL=10
elapsed=0
while (( elapsed < POLL_TIMEOUT )); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Pending")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled)
      break
      ;;
    *)
      sleep "$POLL_INTERVAL"
      elapsed=$(( elapsed + POLL_INTERVAL ))
      ;;
  esac
done

echo ""
echo "=== STDOUT ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardOutputContent' --output text

echo ""
echo "=== STDERR ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardErrorContent' --output text

echo ""
echo "=== Status: $STATUS ==="

[[ "$STATUS" == "Success" ]]

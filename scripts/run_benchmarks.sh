#!/usr/bin/env bash
#
# Run trnrand benchmarks on the trnrand CI instance and pull results back.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_benchmarks.sh [instance_type]
#
# Default instance_type is trn1. Produces:
#   - benchmark_results.json (raw pytest-benchmark output)
#
# This script:
#   1. Starts the tagged instance (if stopped)
#   2. Waits for SSM agent
#   3. Runs `pytest benchmarks/ --benchmark-only --benchmark-json=...`
#   4. Pulls the JSON back via SSM (cat + base64)
#   5. Stops the instance (always, even on failure)

set -euo pipefail

INSTANCE_TYPE="${1:-trn1}"
TAG="trnrand-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"
LOCAL_OUT="${BENCH_OUTPUT:-benchmark_results.json}"

: "${AWS_PROFILE:?Set AWS_PROFILE, e.g. AWS_PROFILE=aws ./scripts/run_benchmarks.sh}"

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,stopping,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No instance found with Name=$TAG" >&2
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
if [[ "$STATE" == "stopping" ]]; then
  echo "Instance is stopping; waiting..."
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
  STATE="stopped"
fi
if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi
echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "Waiting for SSM agent..."
for i in $(seq 1 30); do
  STATUS=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' \
    --output text 2>/dev/null || echo "None")
  if [[ "$STATUS" == "Online" ]]; then echo "SSM agent online."; break; fi
  if [[ $i -eq 30 ]]; then echo "ERROR: SSM agent not online" >&2; exit 1; fi
  sleep 10
done

REMOTE_JSON="/tmp/trnrand_bench.json"

echo "Sending benchmark command (SHA=$SHA)..."
BENCH_SCRIPT="source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnrand && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  pytest benchmarks/bench_rand.py benchmarks/bench_streaming.py --benchmark-only --benchmark-json=$REMOTE_JSON --tb=short 2>&1 | tail -120 && \
  echo BENCH_OK"

CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnrand benchmarks @ $SHA" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$BENCH_SCRIPT\\\"\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for benchmarks to complete (this may take 10+ minutes)..."

STATUS="InProgress"
for i in $(seq 1 80); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 30
done

echo ""
echo "=== Run status: $STATUS ==="
echo "=== STDOUT (tail) ==="
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text

if [[ "$STATUS" != "Success" ]]; then
  echo "=== STDERR (tail) ==="
  aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text
  exit 1
fi

echo ""
echo "Pulling results JSON via SSM (base64)..."
FETCH_CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "{\"commands\":[\"base64 -w0 $REMOTE_JSON\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

# Short poll for the fetch
for i in $(seq 1 12); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$FETCH_CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 5
done

if [[ "$STATUS" != "Success" ]]; then
  echo "ERROR: Could not fetch JSON. Status: $STATUS" >&2
  aws ssm get-command-invocation --command-id "$FETCH_CMD_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardErrorContent' --output text >&2
  exit 1
fi

aws ssm get-command-invocation --command-id "$FETCH_CMD_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query 'StandardOutputContent' --output text \
  | tr -d '\n' | base64 --decode > "$LOCAL_OUT"

echo "Wrote $LOCAL_OUT ($(wc -c < "$LOCAL_OUT") bytes)"
echo ""
echo "Next: scripts/bench_to_md.py $LOCAL_OUT > docs/benchmarks_table.md"

# AWS Setup for Neuron Tests

To run `pytest -m neuron` against real Trainium hardware, we use a local workflow:

- Provision a Trainium EC2 instance with Terraform (stays stopped when not testing)
- Run the test script locally from your machine, using `AWS_PROFILE=aws`
- The script starts the instance, runs pytest via SSM, prints output, stops the instance

GitHub Actions does **not** touch AWS. All AWS interaction is human-initiated.

## One-time setup

### 1. Provision the CI instance

Pick a VPC + subnet in a region with trn1/trn2/inf2 capacity. `trn1.2xlarge` is cheapest for basic validation.

```bash
cd infra/terraform

AWS_PROFILE=aws terraform init
AWS_PROFILE=aws terraform apply \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx" \
  -var="instance_type=trn1.2xlarge"
```

Capture `instance_id` from the outputs. User-data takes ~5 minutes to install the Neuron SDK and clone trnrand.

Stop the instance once ready:

```bash
AWS_PROFILE=aws aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)
```

## Running neuron tests

```bash
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh
# or for trn2 / inf2:
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh trn2
```

The script will:

1. Look up the tagged instance (`Name=trnrand-ci-trn1` by default)
2. Start it if stopped; wait for SSM agent
3. Send the pytest command over SSM
4. Print stdout/stderr
5. **Stop the instance in a trap** (even if pytest fails or you Ctrl-C)

It exits non-zero if any test fails.

## Running the DF-MP2 bench

Same instance, same SSM mechanism, runs `examples/df_mp2.py --bench` to
capture per-step timing across small / medium / large synthetic shapes:

```bash
AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh                 # all 3 shapes
AWS_PROFILE=aws ./scripts/run_df_mp2_bench.sh --shape medium  # one shape
```

Each shape runs cold then warm in the same Python process, so NEFF cache
effects are visible in the reported numbers.

## Cost

Stopped = EBS only (~$10/mo for 100 GB gp3). Running:

| Type | Hourly | Typical run (10 min) |
|------|-------:|---------------------:|
| trn1.2xlarge | $1.34 | $0.22 |
| trn2.8xlarge | $10.00 | $1.67 |
| inf2.xlarge | $0.76 | $0.13 |

## Troubleshooting

**"No instance found with Name=trnrand-ci-trn1"**
— Run `terraform apply` first, or check that the tag matches.

**SSM `InvalidInstanceId` error**
— Instance hasn't finished booting/registering. Wait 1-2 minutes and retry.

**User-data didn't finish (`neuronxcc not found`)**
— SSH in via SSM session and re-run manually:
```bash
aws ssm start-session --target $INSTANCE_ID
cd /home/ubuntu/trnrand && pip install -e '.[neuron,dev]'
```

**`InsufficientInstanceCapacity` when starting the instance**
— AWS may temporarily be out of Trainium in that AZ. Wait and retry, or re-provision in a different AZ.

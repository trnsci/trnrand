# AWS Setup for Neuron Tests

To run `pytest -m neuron` against real Trainium hardware, we use a local workflow:

- Provision a Trainium EC2 instance with Terraform (stays stopped when not testing)
- Run the test script locally from your machine, using `AWS_PROFILE=aws`
- The script starts the instance, runs pytest via SSM, prints output, stops the instance

GitHub Actions does **not** touch AWS. All AWS interaction is human-initiated.

## One-time setup

### 1. Provision the CI instance

Two separate Terraform roots, one per hardware family:

| Hardware | Terraform root | Default region | Instance |
|---|---|---|---|
| Trainium1 | `infra/terraform/` | `us-east-1` | `trn1.2xlarge` |
| Trainium2 | `infra/terraform-trn2/` | `sa-east-1` | `trn2.3xlarge` |

**trn2 availability (as of 2026-04-16):**

| Instance type | Region | AZs |
|---|---|---|
| trn2.xlarge | — | not yet offered |
| trn2.3xlarge | sa-east-1 | a, b, c |
| trn2.48xlarge | us-east-2 | a, b, c |

**Trainium1 (trn1) — us-east-1:**
```bash
cd infra/terraform
AWS_PROFILE=aws terraform init
AWS_PROFILE=aws terraform apply \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx"
```

**Trainium2 (trn2) — sa-east-1:**
```bash
cd infra/terraform-trn2
AWS_PROFILE=aws terraform init
AWS_PROFILE=aws terraform apply \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx"
```

You'll need a VPC and subnet in the target region. User-data takes ~5 minutes to install the Neuron SDK and clone trnrand.

Stop the instance once ready:

```bash
# trn1
cd infra/terraform
AWS_PROFILE=aws aws ec2 stop-instances \
  --instance-ids $(AWS_PROFILE=aws terraform output -raw instance_id) --region us-east-1

# trn2
cd infra/terraform-trn2
AWS_PROFILE=aws aws ec2 stop-instances \
  --instance-ids $(AWS_PROFILE=aws terraform output -raw instance_id) --region sa-east-1
```

## Running neuron tests

```bash
# trn1 (default)
AWS_PROFILE=aws ./scripts/run_neuron_tests.sh

# trn2 — note AWS_REGION override (instance is in sa-east-1)
AWS_PROFILE=aws AWS_REGION=sa-east-1 ./scripts/run_neuron_tests.sh trn2
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
| trn2.3xlarge | $10.00 | $1.67 |
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

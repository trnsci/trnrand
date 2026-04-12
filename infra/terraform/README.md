# Terraform — trnrand Neuron CI Instance

Provisions a Trainium/Inferentia instance for running `pytest -m neuron`
from your local machine via SSM. The instance stays stopped between runs.

GitHub Actions does not touch AWS — everything is driven locally via
`AWS_PROFILE=aws` and `scripts/run_neuron_tests.sh`.

## What it creates

- **EC2 instance** — `trn1.2xlarge` by default (Deep Learning AMI Neuron, Ubuntu 22.04)
- **IAM instance profile** — gives the instance SSM managed-instance-core permissions
- **Security group** — no inbound, outbound only (SSM uses VPC endpoints or NAT)

## Apply

```bash
cd infra/terraform

AWS_PROFILE=aws terraform init

AWS_PROFILE=aws terraform apply \
  -var="vpc_id=vpc-xxxxxx" \
  -var="subnet_id=subnet-xxxxxx"
```

Outputs include `instance_id` and `instance_tag`. Wait ~5 minutes for
user-data to finish (installs Neuron SDK, clones trnrand), then stop it:

```bash
AWS_PROFILE=aws aws ec2 stop-instances \
  --instance-ids $(terraform output -raw instance_id)
```

## Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `aws_region` | `us-east-1` | Trainium: us-east-1, us-west-2, eu-west-1 |
| `instance_type` | `trn1.2xlarge` | Also: `trn2.8xlarge`, `inf2.xlarge` |
| `instance_tag` | `trnrand-ci-trn1` | Must match `scripts/run_neuron_tests.sh [type]` arg |
| `vpc_id` | (required) | |
| `subnet_id` | (required) | Must be in an AZ with capacity for `instance_type` |

For multiple instance types, apply the module multiple times with different
`instance_type` and `instance_tag`.

## Cost

- Stopped: EBS storage only (~$10/mo for 100 GB gp3)
- Running: see main AWS pricing; `trn1.2xlarge` ≈ $1.34/hr

## Capacity issues

Trainium capacity is AZ-specific and often exhausted in popular AZs. If
apply fails with `InsufficientInstanceCapacity`, try a different subnet
in another AZ in the same region. For us-east-1, `us-east-1f` often has
capacity when `us-east-1a` does not.

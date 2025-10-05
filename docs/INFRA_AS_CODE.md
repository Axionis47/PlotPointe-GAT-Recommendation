# Phase 6: Infra as Code (Terraform)

This phase adds a minimal, safe Terraform skeleton to provision shared infra required by training jobs:

- Artifact bucket (GCS) with versioning, retention, and labels
- IAM binding for the pipeline service account to manage objects
- Required API activations (AI Platform, Storage)

Nothing is applied automatically. CI only runs `terraform fmt` and `terraform validate`.

## Files
- `terraform/versions.tf` — provider constraints
- `terraform/variables.tf` — input variables
- `terraform/main.tf` — resources (bucket, IAM, services)
- `terraform/example.auto.tfvars` — example inputs
- `.github/workflows/terraform.yml` — CI for fmt/validate only

## Prereqs
- Terraform >= 1.5
- GCP Project IAM: Owner or Editor (for initial provisioning) or a set of more granular roles that cover the listed resources
- Auth via Application Default Credentials (`gcloud auth application-default login`) or a service account key exported in your shell

## Usage (manual)

```
cd terraform
cp example.auto.tfvars my.auto.tfvars
# edit values (project_id, artifacts_bucket, service_account_email, etc.)

terraform init
terraform plan -var-file=my.auto.tfvars
# Review plan output carefully
terraform apply -var-file=my.auto.tfvars
```

## Notes
- Bucket names must be globally unique.
- Retention lifecycle is included as a safety net; tune `retention_days` as needed or remove the rule.
- API enablement resources (`google_project_service`) will enable services if not already enabled. They are kept even on destroy (`disable_on_destroy=false`).
- To add more resources later (e.g., BigQuery datasets, additional buckets), extend `main.tf` or split into modules.


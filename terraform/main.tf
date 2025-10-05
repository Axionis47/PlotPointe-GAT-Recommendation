# Terraform IaC for PlotPointe GAT training infra (non-destructive until applied)
# This module focuses on shared infra: artifact bucket and IAM bindings required for jobs.

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required services (only on apply)
resource "google_project_service" "services" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
  ])
  project = var.project_id
  service = each.key
  disable_on_destroy = false
}

# Central artifacts bucket (models, metrics, staging)
resource "google_storage_bucket" "artifacts" {
  name          = var.artifacts_bucket
  location      = var.bucket_location
  storage_class = var.bucket_storage_class
  uniform_bucket_level_access = true
  versioning { enabled = true }
  lifecycle_rule {
    action { type = "Delete" }
    condition { age = var.retention_days }
  }
  labels = merge({
    system = "plotpointe-recsys",
    env    = var.env,
  }, var.labels)
}

# Allow pipeline SA to manage objects in artifacts bucket
resource "google_storage_bucket_iam_member" "artifacts_object_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.service_account_email}"
}

output "artifacts_bucket_name" {
  value       = google_storage_bucket.artifacts.name
  description = "Artifacts bucket name"
}


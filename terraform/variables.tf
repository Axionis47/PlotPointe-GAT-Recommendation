variable "project_id" {
  description = "GCP project id"
  type        = string
}

variable "region" {
  description = "Region for Vertex/Storage"
  type        = string
  default     = "us-central1"
}

variable "env" {
  description = "Environment label"
  type        = string
  default     = "dev"
}

variable "labels" {
  description = "Additional resource labels"
  type        = map(string)
  default     = {}
}

variable "service_account_email" {
  description = "Service account used by Vertex jobs (e.g., sa-pipeline@<project>.iam.gserviceaccount.com)"
  type        = string
}

variable "artifacts_bucket" {
  description = "Artifacts bucket name (must be globally unique)"
  type        = string
}

variable "bucket_location" {
  description = "GCS bucket location"
  type        = string
  default     = "US"
}

variable "bucket_storage_class" {
  description = "GCS storage class"
  type        = string
  default     = "STANDARD"
}

variable "retention_days" {
  description = "Retention policy in days (objects older than this may be deleted)"
  type        = number
  default     = 365
}


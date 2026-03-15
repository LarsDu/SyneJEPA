terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Service account for Vertex AI training
resource "google_service_account" "vertex_training" {
  account_id   = "synejepa-training"
  display_name = "SyneJEPA Vertex AI Training"
}

# IAM: Vertex AI user
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_training.email}"
}

# IAM: Storage access for checkpoints
resource "google_project_iam_member" "storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.vertex_training.email}"
}

# IAM: Artifact Registry for container images
resource "google_project_iam_member" "artifact_registry_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.vertex_training.email}"
}

# Storage bucket for training checkpoints and logs
resource "google_storage_bucket" "checkpoints" {
  name          = "${var.project_id}-synejepa-checkpoints"
  location      = var.region
  force_destroy = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Artifact Registry for training container images
resource "google_artifact_registry_repository" "training_images" {
  location      = var.region
  repository_id = "synejepa-training"
  description   = "Container images for SyneJEPA training"
  format        = "DOCKER"
}

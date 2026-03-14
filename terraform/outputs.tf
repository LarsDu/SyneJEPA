output "training_service_account_email" {
  description = "Service account for Vertex AI training jobs"
  value       = google_service_account.vertex_training.email
}

output "checkpoint_bucket" {
  description = "GCS bucket for training checkpoints"
  value       = google_storage_bucket.checkpoints.name
}

output "artifact_registry_url" {
  description = "Docker registry URL for training images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.training_images.repository_id}"
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "gpu_type" {
  description = "GPU accelerator type for training"
  type        = string
  default     = "NVIDIA_TESLA_A100"
}

variable "gpu_count" {
  description = "Number of GPUs per training node"
  type        = number
  default     = 4
}

#!/usr/bin/env bash
# One-time Google Cloud setup so GitHub Actions can deploy the API to Cloud Run with no stored keys.
#
# Run it in Cloud Shell (https://shell.cloud.google.com), where gcloud is installed and signed in:
#   PROJECT_ID=your-project GITHUB_REPO=your-github-user/your-repo bash deploy/gcp_setup.sh
#
# Safe to re-run: each resource is created only if it does not exist yet.
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project ID}"
: "${GITHUB_REPO:?Set GITHUB_REPO to owner/name of your GitHub repository}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-yt-sentiment}"
SERVICE="${SERVICE:-yt-sentiment-api}"
DVC_BUCKET="${DVC_BUCKET:-${PROJECT_ID}-dvc}"
SECRET="${SECRET:-youtube-api-key}"
POOL="github"
PROVIDER="yt-sentiment-repo"

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
DEPLOYER_SA="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_SA="yt-sentiment-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}"

exists() { "$@" >/dev/null 2>&1; }

echo "== Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com storage.googleapis.com

echo "== Artifact Registry repository: $AR_REPO"
exists gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" || gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION" --description="YouTube sentiment API images"

echo "== Cloud Storage bucket for the DVC remote: gs://$DVC_BUCKET"
exists gcloud storage buckets describe "gs://$DVC_BUCKET" || gcloud storage buckets create "gs://$DVC_BUCKET" --location="$REGION" --uniform-bucket-level-access

echo "== Service accounts"
exists gcloud iam service-accounts describe "$RUNTIME_SA" || gcloud iam service-accounts create yt-sentiment-runtime --display-name="YouTube sentiment API (runtime)"
exists gcloud iam service-accounts describe "$DEPLOYER_SA" || gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"

echo "== YouTube API key in Secret Manager: $SECRET"
if ! exists gcloud secrets describe "$SECRET"; then
  gcloud secrets create "$SECRET" --replication-policy=automatic
  read -rsp "Paste your YouTube Data API key (input hidden), then press Enter: " YOUTUBE_KEY
  echo
  printf '%s' "$YOUTUBE_KEY" | gcloud secrets versions add "$SECRET" --data-file=-
  unset YOUTUBE_KEY
fi

echo "== Runtime account may read only the YouTube key"
gcloud secrets add-iam-policy-binding "$SECRET" --member="serviceAccount:$RUNTIME_SA" --role=roles/secretmanager.secretAccessor --quiet >/dev/null

echo "== Deployer account: deploy to Cloud Run, push images, read DVC data, run the service as the runtime account"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$DEPLOYER_SA" --role=roles/run.admin --condition=None --quiet >/dev/null
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" --location="$REGION" --member="serviceAccount:$DEPLOYER_SA" --role=roles/artifactregistry.writer --quiet >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://$DVC_BUCKET" --member="serviceAccount:$DEPLOYER_SA" --role=roles/storage.objectViewer --quiet >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member="serviceAccount:$DEPLOYER_SA" --role=roles/iam.serviceAccountUser --quiet >/dev/null

echo "== Workload Identity Federation: only $GITHUB_REPO may act as the deployer"
exists gcloud iam workload-identity-pools describe "$POOL" --location=global || gcloud iam workload-identity-pools create "$POOL" --location=global --display-name="GitHub Actions"
exists gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global --workload-identity-pool="$POOL" || gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global --workload-identity-pool="$POOL" --display-name="YouTube sentiment repo" --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" --attribute-condition="assertion.repository == '${GITHUB_REPO}'" --issuer-uri="https://token.actions.githubusercontent.com"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" --role=roles/iam.workloadIdentityUser --quiet >/dev/null

cat <<EOF

Setup complete. Add these as GitHub repository variables
(Settings > Secrets and variables > Actions > Variables tab; none of them are secret):

  GCP_PROJECT_ID      $PROJECT_ID
  GCP_REGION          $REGION
  GCP_WIF_PROVIDER    ${POOL_ID}/providers/${PROVIDER}
  GCP_DEPLOYER_SA     $DEPLOYER_SA
  GCP_RUNTIME_SA      $RUNTIME_SA
  GCP_AR_REPO         $AR_REPO
  CLOUD_RUN_SERVICE   $SERVICE
  YOUTUBE_KEY_SECRET  $SECRET

Then, on your own machine, point DVC at the bucket and upload the trained model:

  dvc remote add -d -f storage gs://$DVC_BUCKET/dvc
  dvc push
EOF

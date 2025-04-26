#!/bin/bash
set -euo pipefail

echo "Container started. Waiting for manual setup."

# Make sure git and curl are installed so we can fetch the provision script
apt-get update && apt-get install -y git curl

# Download your provisioning script from GitHub
curl -o /root/provision.sh https://raw.githubusercontent.com/zachmccrth/SparseMate/refs/heads/main/provision.sh
chmod +x /root/provision.sh

echo "Provisioning script downloaded to /root/provision.sh"
echo "When ready, run: bash /root/provision.sh"

# Keep container alive waiting for you
tail -f /dev/null

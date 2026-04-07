#!/bin/bash
# ============================================================
# aws_setup.sh
# Launch an A100 spot instance on AWS for EgoBlind-RA training
# ============================================================
#
# BEFORE RUNNING:
# 1. Install AWS CLI: pip install awscli && aws configure
# 2. Request a quota increase for p4d.24xlarge spot instances
#    in your preferred region (us-east-1 or us-west-2 are cheapest).
#    Go to: Service Quotas > EC2 > "Running On-Demand P instances"
#    and "All P Spot Instance Requests"
#
# COST ESTIMATES (as of early 2026):
#   p4d.24xlarge (8x A100-40GB): ~$10-12/hr spot (~$32/hr on-demand)
#   p3.2xlarge   (1x V100-16GB): ~$0.90/hr spot  (tight but possible for A3B)
#   g5.2xlarge   (1x A10G-24GB): ~$0.75/hr spot  (may work with 4-bit quant)
#
# RECOMMENDED: g5.12xlarge (4x A10G, ~$3.50/hr spot)
#   - Fits Kimi-VL-A3B LoRA training comfortably
#   - $40 budget = ~11 hours, plenty for SFT + DPO
#
# ALTERNATIVE: Use Lambda Labs or RunPod for simpler setup
#   - Lambda: 1x A100-80GB at ~$1.50/hr (no spot, just pay-as-you-go)
#   - RunPod: 1x A100-80GB at ~$1.64/hr (community cloud, cheaper)
#   These are much simpler than AWS if you just need a GPU.
# ============================================================

set -e

# --- CONFIG ---
INSTANCE_TYPE="g5.12xlarge"      # 4x A10G-24GB, good price/perf
AMI_ID="ami-0c7217cdde317cfec"   # Ubuntu 22.04 (us-east-1, update for your region)
KEY_NAME="your-key-pair-name"    # Your SSH key pair name
SECURITY_GROUP="sg-xxxxxxxx"     # Security group with SSH (port 22) open
REGION="us-east-1"
SPOT_PRICE="5.00"                # Max spot bid (actual price is usually much lower)

echo "=== Launching spot instance ==="
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --region $REGION \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'$SPOT_PRICE'","SpotInstanceType":"one-time"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "=== Instance ready ==="
echo "SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "Next: run remote_setup.sh on the instance"

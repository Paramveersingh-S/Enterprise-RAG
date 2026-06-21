# Enterprise RAG AWS Deployment Guide

This guide provides step-by-step instructions for deploying the Enterprise-RAG system to AWS using an EC2 (Elastic Compute Cloud) instance. Because the entire application is containerized with Docker, this process is straightforward and avoids complex cloud infrastructure orchestration.

> [!NOTE]
> This guide sets up a single-node deployment (all containers on one VM). This is ideal for testing, staging, and moderate production workloads. For high-availability enterprise environments, consider migrating to AWS ECS with managed databases.

---

## Step 1: Launch an EC2 Instance

1. Log in to your **[AWS Management Console](https://console.aws.amazon.com/)** and navigate to **EC2**.
2. Click **Launch Instance**.
3. **Name**: Enter `Enterprise-RAG-Server`
4. **OS Images (AMI)**: Select **Ubuntu** (Ubuntu Server 22.04 LTS).
5. **Instance Type**: Select **`t3.large`** or **`t3.xlarge`** (The system requires at least 4-8GB of RAM to comfortably run Qdrant, Neo4j, Redis, and the Python workers).
6. **Key Pair**: Create a new key pair (RSA, `.pem`) and download it to your local machine. You will need this to SSH into the server.
7. **Network Settings**:
   - Check **Allow SSH traffic from Anywhere** (or your specific IP for better security).
   - Check **Allow HTTP traffic from the internet**.
8. **Storage**: Set the root volume to at least **30 GB** (gp3) to ensure enough space for Docker images and database volumes.
9. Click **Launch Instance**.

---

## Step 2: Configure Security Groups (Firewall)

The FastAPI server runs on port 8000. We need to open this port so you can access the API.

1. Go to your EC2 instance details and click on the **Security** tab.
2. Click on the attached **Security Group** link.
3. Click **Edit inbound rules** > **Add rule**.
4. Set **Type** to `Custom TCP`, **Port range** to `8000`, and **Source** to `Anywhere-IPv4` (`0.0.0.0/0`).
5. Click **Save rules**.

---

## Step 3: Connect to Your Server

Open your local terminal (Command Prompt, PowerShell, or Mac/Linux terminal) and SSH into your new server using the `.pem` key you downloaded:

```bash
# Set secure permissions on your key (Mac/Linux only)
chmod 400 your-key.pem

# Connect to the server (replace the IP with your EC2 Public IPv4 address)
ssh -i "your-key.pem" ubuntu@<YOUR_EC2_PUBLIC_IP>
```

---

## Step 4: Install Docker and Git

Once you are connected to the Ubuntu server terminal, run these commands to install Docker and Git:

```bash
# Update package lists
sudo apt-get update

# Install Git
sudo apt-get install -y git

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install -y docker-compose-plugin

# Give the 'ubuntu' user permission to run Docker without sudo
sudo usermod -aG docker ubuntu

# Apply the new permissions (or log out and log back in)
newgrp docker
```

---

## Step 5: Clone the Repository

Clone your project from GitHub onto the server:

```bash
git clone https://github.com/Paramveersingh-S/Enterprise-RAG.git
cd Enterprise-RAG
```

---

## Step 6: Configure the `.env` File

Since your `.env` file is hidden and ignored by Git (for security), you must recreate it on the AWS server.

```bash
nano .env
```

Paste your exact configuration into the file:

```env
# GROQ SETTINGS
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.0
GROQ_MAX_TOKENS=2048

# HUGGINGFACE SETTINGS
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# INTERNAL DOCKER SETTINGS
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=enterprise_docs
QDRANT_VECTOR_SIZE=1024
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://redis:6379/0
ENVIRONMENT=production
```

Save and exit nano (`Ctrl+O`, `Enter`, `Ctrl+X`).

---

## Step 7: Launch the Enterprise RAG System

You are now ready to build and start the entire stack.

```bash
docker compose up -d --build
```

This will take a few minutes as it pulls the database images and builds the Python environment. You can check the status by running:
```bash
docker compose ps
```

---

## Step 8: Test the Live Cloud API

Your system is now live on the internet! From your own local computer, you can test it by sending a cURL request to your EC2 instance's Public IP address.

### 1. Ingest a Document
```bash
curl -X POST http://<YOUR_EC2_PUBLIC_IP>:8000/api/v1/ingest \
     -H "Content-Type: application/json" \
     -d '{"urls": ["https://en.wikipedia.org/wiki/Contract"]}'
```

### 2. Ask a Question
```bash
curl -X POST http://<YOUR_EC2_PUBLIC_IP>:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the core elements of a valid contract?"}'
```

> [!TIP]
> **Next Steps for Production**: 
> 1. Set up a domain name (Route 53) pointing to your IP.
> 2. Use an Nginx reverse proxy and Let's Encrypt to enable HTTPS (SSL/TLS).
> 3. Secure your Neo4j and Qdrant instances by removing external port mappings in `docker-compose.yml` if they don't need public access.

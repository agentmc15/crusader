#!/bin/bash

echo "🚀 Starting Agentic Peer-Review 2.0 System..."

# Check if we're in the right directory
if [[ ! -f "README.md" ]] || [[ ! -d "infra" ]]; then
    echo "❌ Please run this script from the root of the crusader repository"
    exit 1
fi

# Navigate to infrastructure directory and start services
cd infra
echo "🔧 Starting services with Docker Compose..."
docker-compose up -d --build

echo "⏳ Waiting for services to start..."
sleep 30

echo ""
echo "✅ APR 2.0 is now running!"
echo ""
echo "🌐 Access points:"
echo "   Submission API:        http://localhost:8000"
echo "   Orchestrator:          http://localhost:8001"
echo "   Plagiarism Detector:   http://localhost:8002"
echo "   Methodology Reviewer:  http://localhost:8003"
echo "   Ethics Agent:          http://localhost:8004"
echo "   Database:              localhost:5432"
echo "   Redis:                 localhost:6379"
echo ""
echo "🔧 Useful commands:"
echo "   View logs: cd infra && docker-compose logs"
echo "   Stop system: cd infra && docker-compose down"
echo "   Reset system: cd infra && docker-compose down -v && docker-compose up -d"

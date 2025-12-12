#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📦 prview Publishing Script${NC}"
echo "================================"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}❌ Error: You have uncommitted changes${NC}"
    echo "Please commit or stash your changes first"
    exit 1
fi

# Show current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "\n${YELLOW}Current branch: ${BRANCH}${NC}"

# Get current version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo -e "${YELLOW}Current version: ${VERSION}${NC}"

# Check if tag already exists
if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Tag v${VERSION} already exists${NC}"
    echo "Please bump the version in pyproject.toml and prview/__init__.py"
    exit 1
fi

# Confirm publication
echo -e "\n${YELLOW}This will:${NC}"
echo "1. Clean previous builds"
echo "2. Build new package"
echo "3. Upload to PyPI"
echo "4. Create git tag v${VERSION}"
echo "5. Push to GitHub"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted${NC}"
    exit 0
fi

# Clean previous builds
echo -e "\n${GREEN}🧹 Cleaning previous builds...${NC}"
rm -rf dist/ build/ *.egg-info prview.egg-info

# Build package
echo -e "\n${GREEN}🔨 Building package...${NC}"
python -m build

# Check if PyPI credentials are configured
if [ ! -f ~/.pypirc ]; then
    echo -e "\n${YELLOW}⚠️  No ~/.pypirc found${NC}"
    echo "You'll need to enter your PyPI credentials"
    echo "Username: __token__"
    echo "Password: <your-pypi-token>"
fi

# Upload to PyPI
echo -e "\n${GREEN}📤 Uploading to PyPI...${NC}"
python -m twine upload dist/*

if [ $? -ne 0 ]; then
    echo -e "\n${RED}❌ Upload failed${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ Successfully published to PyPI!${NC}"

# Create git tag
echo -e "\n${GREEN}🏷️  Creating git tag v${VERSION}...${NC}"
git tag -a "v${VERSION}" -m "Release v${VERSION}"

# Push everything
echo -e "\n${GREEN}⬆️  Pushing to GitHub...${NC}"
git push origin ${BRANCH}
git push origin "v${VERSION}"

echo -e "\n${GREEN}✅ Publishing complete!${NC}"
echo -e "\n${YELLOW}Users can now install with:${NC}"
echo "  pip install prview"
echo "  pip install 'prview[web]'  # for web UI"
echo "  pip install 'prview[all]'  # for all features"
echo ""
echo -e "${YELLOW}View on PyPI:${NC}"
echo "  https://pypi.org/project/prview/${VERSION}/"

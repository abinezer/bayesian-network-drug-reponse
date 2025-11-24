# GitHub Repository Setup Instructions

## Step 1: Initialize Git Repository

```bash
cd /cellar/users/abishai/ClassProjects/cse250A
git init
```

## Step 2: Add All Files

```bash
git add .
```

## Step 3: Make Initial Commit

```bash
git commit -m "Initial commit: Bayesian Network for Palbociclib drug response prediction"
```

## Step 4: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `cse250A-palbociclib-bayesian-network` (or your preferred name)
3. Description: "Bayesian Network for predicting Palbociclib drug response using gene alterations and pathway activations"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 5: Add Remote and Push

After creating the repo on GitHub, GitHub will show you commands. Use these:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/cse250A-palbociclib-bayesian-network.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/cse250A-palbociclib-bayesian-network.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Alternative: All Commands in One Block

```bash
cd /cellar/users/abishai/ClassProjects/cse250A
git init
git add .
git commit -m "Initial commit: Bayesian Network for Palbociclib drug response prediction"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

## Notes

- Replace `YOUR_USERNAME` with your GitHub username
- Replace `REPO_NAME` with your chosen repository name
- If you get authentication errors, you may need to set up a Personal Access Token (Settings > Developer settings > Personal access tokens)


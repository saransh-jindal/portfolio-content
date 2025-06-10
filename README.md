# Portfolio Content Repository Structure

## Overview

This repository contains the content that automatically populates your portfolio website. When you add markdown files to specific folders, they will appear as cards on your website within minutes.

## Repository Structure

Create a GitHub repository named `portfolio-content` with this exact structure:

```
portfolio-content/
├── featured-projects/
│   ├── agri-futures/
│   │   └── README.md
│   ├── stochastic-volatility-modeling-research/
│   │   └── README.md
│   ├── mathematical-risk-analytics-engine/
│   │   └── README.md
│   └── cryptocurrency-market-analysis/
│       └── README.md
├── research-insights/
│   ├── the-mathematics-behind-market-volatility-clustering/
│   │   └── README.md
│   ├── my-take-on-the-2024-crypto-market-a-statistical-perspective/
│   │   └── README.md
│   └── itos-lemma-the-foundation-of-modern-finance/
│       └── README.md
└── README.md
```

## Quick Setup Instructions

1. **Create the repository**:
   ```bash
   # On GitHub, create a new repository named "portfolio-content"
   git clone https://github.com/saransh-jindal/portfolio-content.git
   cd portfolio-content
   ```

2. **Copy the content files**:
   ```bash
   # Copy all the content from your local quant-portfolio/github-content/ folder
   # to this repository
   ```

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Initial portfolio content"
   git push origin main
   ```

## Content Format

### For Projects (featured-projects/)

Each project should be in its own folder with a `README.md` file containing:

```markdown
---
title: "Your Project Name"
description: "Brief description"
technologies: ["Python", "TensorFlow", "NumPy"]
specialties: ["Machine Learning", "Risk Analytics"]
metrics: {
  "accuracy": "92%",
  "performance": "15.2%"
}
status: "Completed"
year: "2024"
featured: true
---

# Your Project Documentation
...
```

### For Articles (research-insights/)

Each article should be in its own folder with a `README.md` file containing:

```markdown
---
title: "Article Title"
excerpt: "Brief excerpt for the card"
category: "Mathematical Finance"
readTime: 8
publishDate: "2024-12-15"
featured: true
tags: ["Stochastic Calculus", "Options Pricing"]
---

# Your Article Content
...
```

## How It Works

1. **Automatic Detection**: Your website checks these folders every time someone visits
2. **Metadata Parsing**: The frontmatter (content between `---` lines) becomes card data
3. **Content Display**: The markdown content is available when users click through
4. **Live Updates**: Changes appear on your website immediately after pushing to GitHub

## Adding New Content

### New Project

1. Create a new folder in `featured-projects/`
2. Add a `README.md` file with the proper frontmatter
3. Commit and push to GitHub
4. Your website will automatically show the new project card

### New Article

1. Create a new folder in `research-insights/`
2. Add a `README.md` file with the proper frontmatter
3. Commit and push to GitHub
4. Your website will automatically show the new article card

## Tips

- Use descriptive folder names (lowercase, hyphens instead of spaces)
- Keep excerpts under 200 characters
- Include relevant tags for better categorization
- Set `featured: true` for your best 2-3 projects/articles
- Use realistic metrics and data
- Include proper contact information and links

## File Names to Copy

From your local `quant-portfolio/github-content/` folder, copy these files:

### Featured Projects
- `featured-projects/agri-futures/README.md`
- `featured-projects/stochastic-volatility-modeling-research/README.md`
- `featured-projects/mathematical-risk-analytics-engine/README.md`
- `featured-projects/cryptocurrency-market-analysis/README.md`

### Research Insights
- `research-insights/the-mathematics-behind-market-volatility-clustering/README.md`
- `research-insights/my-take-on-the-2024-crypto-market-a-statistical-perspective/README.md`

## Your Website URLs

Once you create this repository, your portfolio will automatically link to:
- Project cards → `https://github.com/saransh-jindal/portfolio-content/tree/main/featured-projects/[project-folder]`
- Article cards → `https://github.com/saransh-jindal/portfolio-content/tree/main/research-insights/[article-folder]`

## Need Help?

If the content isn't showing up:
1. Check that folder names match the URL pattern (lowercase, hyphens)
2. Ensure the frontmatter syntax is correct (YAML format)
3. Verify the repository is public
4. Wait a few minutes for caching to refresh

Your portfolio website will automatically stay updated with your latest work!

---
layout: post
title: How to setup this site yourself
summary: Setting up this blog using Jekyll
tags: [blog]
usemathjax: true
---

This site was built using [Jekyll](https://jekyllrb.com/docs/). You might want to check the setup guide to build something more accustomed to your needs.

However, if you're curious about this specific one (e.g. future me)

```bash
git clone https://github.com/manuelmazzuola/asko # forked: https://github.com/panpan2/asko
cd asko
bundle install
# Update kramdown to 2.3.1
gem install kramdown
bundle lock --update
bundle exec jekyll serve
// Enlarge margin
// add latex support
// add a new post
// write posts :)
```

$$\mathbf{E}=mc^2$$

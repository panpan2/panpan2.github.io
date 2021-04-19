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
# Setup mathjax support
# On the top of the post add 'usemathjax: true'
# Add the following to '_includes/head.html'
'''
<!-- for mathjax support -->
    {% if page.usemathjax %}
      <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
        TeX: { equationNumbers: { autoNumber: "AMS" } }
        });
      </script>
      <script type="text/javascript" async src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    {% endif %}
'''
# Render LaTeX $$E=mc^2$$
bundle exec jekyll serve
// Enlarge margin
// add latex support
// add a new post
// write posts :)
```

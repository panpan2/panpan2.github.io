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
# Setup mathjax support:
# 1: On the top of each post add 'usemathjax: true'
# 2: Add the following to '_includes/head.html'
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
# 3: Render LaTeX $$E=mc^2$$
# To build this
bundle exec jekyll serve
# Add a new post
touch _posts/2021-04-19-Another-one.md
# Write more posts :)
```

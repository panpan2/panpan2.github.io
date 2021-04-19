---
layout: post
title: How to setup this site yourself
summary: Setting up this blog using Jekyll
tags: [blog]
usemathjax: true
comments: true
---

This site was built using [Jekyll](https://jekyllrb.com/docs/). You might want to check the setup guide to build something more accustomed to your needs.

However, if you're curious about this specific one (e.g. future me)

<br />

## Theme setup

```bash
git clone https://github.com/manuelmazzuola/asko # forked: https://github.com/panpan2/asko
cd asko
bundle install
# Update kramdown to 2.3.1
gem install kramdown
bundle lock --update
```

<br />

## MathJax support

On the top of each post add `usemathjax: true`

Now add the following to `_includes/head.html`:

```
{% raw %}
  <!-- for mathjax support -->
  {% if page.usemathjax %}
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
    });
  </script>
  <script
    type="text/javascript"
    async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  ></script>
  {% endif %}
{% endraw %}
```

Render LaTex `$$E=mc^2$$`: $$E=mc^2$$

<br />

## Let's build this

```
bundle exec jekyll serve
```

<br />

## Write, write, write

```
# Add a new post
touch _posts/2021-04-19-Another-one.md
# Write more posts :)
```

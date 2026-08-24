# jaleite.com

Personal website for João A. Leite.

The site is plain HTML, CSS, and JavaScript. GitHub Pages publishes the contents
of `site/` directly, including the images under `site/assets/img/`.

## Local preview

```sh
python3 -m http.server --directory site 4174
```

Then open <http://127.0.0.1:4174/>.

## Code blocks

Writing pages load a local copy of Prism 1.30.0. Add a `language-*` class to
each code block:

```html
<pre><code class="language-python">def greet(name):
    print(f"Hello, {name}")
</code></pre>
```

HTML, CSS, JavaScript, Python, and Bash are bundled. Escape HTML-sensitive
characters inside code, such as `<` as `&lt;`, `>` as `&gt;`, and `&` as
`&amp;`.

## Deployment

Pushes to `main` that change `site/` or the deployment workflow trigger
`.github/workflows/deploy.yml`. The workflow deploys `site/` to GitHub Pages at
<https://jaleite.com/>.

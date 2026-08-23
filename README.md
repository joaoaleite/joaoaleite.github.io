# jaleite.com

Personal website for João A. Leite.

The site is plain HTML and CSS. GitHub Pages publishes the contents of `site/`
directly, including the images under `site/assets/img/`.

## Local preview

```sh
python3 -m http.server --directory site 4174
```

Then open <http://127.0.0.1:4174/>.

## Deployment

Pushes to `main` that change `site/` or the deployment workflow trigger
`.github/workflows/deploy.yml`. The workflow deploys `site/` to GitHub Pages at
<https://jaleite.com/>.

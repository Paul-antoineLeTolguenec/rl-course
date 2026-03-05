# Reinforcement Learning Course

**Paul-Antoine LE TOLGUENEC**

Site de cours interactif déployé sur GitHub Pages.

## Navigation

| Touche | Action |
|--------|--------|
| `↓` / molette bas | Section suivante (Landing → Slides → Notebooks) |
| `↑` / molette haut | Section précédente |
| `→` | Slide suivant |
| `←` | Slide précédent |

## Déploiement

Source GitHub Pages : **GitHub Actions**

## Ajouter des slides

Dans `index.html`, ajouter un bloc `<section>` dans la section slides :

```html
<section>
  <div class="slide-panel">
    <h2>Titre du slide</h2>
    <ul>
      <li>Point clé</li>
    </ul>
  </div>
  <div class="slide-nav">← → naviguer les slides · ↓ ressources</div>
</section>
```

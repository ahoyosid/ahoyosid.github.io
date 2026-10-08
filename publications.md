---
title: Publications
description: >-
    Publications by Andrés Hoyos-Idrobo on learning to rank, optimal transport,
    causal inference and neuroimaging, with summaries, figures and BibTeX.
nav_order: 3
permalink: /publications/
---

Citation counts are on my
[Google Scholar](https://scholar.google.com/citations?user=J3344dQAAAAJ) profile.

{% assign by_year = site.data.publications | group_by: "year" %}
{% for group in by_year %}
<h2>{{ group.name }}</h2>
<ul class="publications">
{% for pub in group.items %}{% include publication.html pub=pub details=true %}
{% endfor %}
</ul>
{% endfor %}

<h2>PhD thesis</h2>
<ul class="publications">
<li class="publication">
    <strong>A. Hoyos-Idrobo</strong>.
    <a href="https://theses.hal.science/tel-01526693" target="_blank">Ensembles of models in fMRI: stable learning in large-scale settings</a>.
    <em>Université Paris-Saclay</em>, 2017.
    {% include bibtex.html key="hoyosidrobo2017ensembles" %}
</li>
</ul>

<script>
    // Copy a paper's BibTeX entry to the clipboard
    document.querySelectorAll('.copy-bibtex').forEach(function (button) {
        button.addEventListener('click', function () {
            var text = button.parentNode.querySelector('pre').textContent;
            navigator.clipboard.writeText(text).then(function () {
                button.textContent = 'Copied';
                setTimeout(function () { button.textContent = 'Copy'; }, 1500);
            });
        });
    });
</script>

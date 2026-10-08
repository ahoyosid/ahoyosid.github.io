---
title: Publications
nav_order: 3
permalink: /publications/
---

The full list, with citations, is on my
[Google Scholar](https://scholar.google.com/citations?user=J3344dQAAAAJ) profile.

{% assign by_year = site.data.publications | group_by: "year" %}
{% for group in by_year %}
<h2>{{ group.name }}</h2>
<ul class="publications">
{% for pub in group.items %}{% include publication.html pub=pub %}
{% endfor %}
</ul>
{% endfor %}

<h2>PhD thesis</h2>
<ul class="publications">
<li class="publication media">
    <div class="media-left">
        <a href="{{ '/img/papers/thesis-ensembles-fmri.svg' | prepend: site.baseurl }}" target="_blank">
            <img class="pub-figure" src="{{ '/img/papers/thesis-ensembles-fmri.svg' | prepend: site.baseurl }}" alt="PhD thesis: ensembles of models for stable, scalable learning on fMRI data" width="150" height="113" loading="lazy">
        </a>
    </div>
    <div class="media-body">
    <strong>A. Hoyos-Idrobo</strong>.
    <a href="https://theses.hal.science/tel-01526693" target="_blank">Ensembles of models in fMRI: stable learning in large-scale settings</a>.
    <em>Université Paris-Saclay</em>, 2017.
    </div>
</li>
</ul>

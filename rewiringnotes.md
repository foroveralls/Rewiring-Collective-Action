# Revision notes
## Figures
- can add two more figures with explanation
- put the field sensitivety analyses in main text


## other
The asymmetry in local rewiring

The action side ("out-links only") is universal and clearly deliberate — the docstrings frame it as follow/unfollow semantics: you control who you follow, not who follows you. The discovery side (find_2_steps walking both successors and predecessors) is the asymmetric part, and it has both a social reading (awareness travels both ways — a follower's activity exposes you to third parties) and a pragmatic one: on sparse directed networks like DPAH, many nodes have low out-degree, so successor-only 2-hop pools would frequently be empty and local rewiring would stall.

Is it only that algorithm? Explicitly coded, yes. But in effect, no: node2vec_cpp.py writes bare u v edgelist lines and invokes the SNAP binary without its directed flag, so node2vec's walks treat the graph as undirected — its discovery is equally direction-blind, just implicitly. WTF is the only algorithm whose discovery respects edge direction; bridge and random don't traverse the graph at all, so the question doesn't arise for them. Worth knowing before you write the Methods bullet, since "discovery is bidirectional" honestly describes two of the five algorithms.
//! Information-set merging by the root decider's observation (docs/search.md 3.16; `Tree::set_merge`, off by
//! default): single-observer information-set MCTS inside the tree (Cowling, Powley & Whitehouse 2012,
//! "Information Set Monte Carlo Tree Search").
//!
//! * **Links.** An edge's children are links: the tree's child signature (chance key, decider, legal options,
//!   module docs of `tree.rs`) and the node it leads to. A walk follows a known link without encoding anything
//!   (by the game contract the same path and chance keys give the root decider the same observation: revelations
//!   are keyed). At a new signature - where the tree of before creates a node - it encodes the reached state from
//!   the viewer's view and looks the observation up in the tree's map: found, a new link to that node (the tree
//!   becomes a DAG); not found, a new node. The key: `obs_hash(encode(viewer))`, the decider and, for the viewer's
//!   own nodes, their options (equal features always had equal options; a world where they differ is kept apart).
//! * **No depth in the key; a path check.** World-dependent auto-steps make depth world-dependent (another seat
//!   with a single option is auto-stepped, one with two decides), so depth cannot be in the key. A walk never holds
//!   a node twice: a link leading to a node already on the walk's path is re-pointed to a fresh unmapped node (an
//!   alias), so path-based backup stays exact.
//! * **Subset-armed nodes of the other seats.** An other seat's node holds the union of the option lists of the
//!   worlds that reached it (appended: indices never change). A walk takes only an option legal in its world:
//!   `argmax Q_a + c P'_a sqrt(max(avail_a, 1)) / (1 + N_a)` over them, P' = the option's mean prior renormalised
//!   over the available ones, avail_a = the walks at the node in which a was legal (counted at selection), FPU the
//!   node's `vec` (with root-view leaf values the viewer's value: the same for every world of the observation).
//!   The viewer's own nodes keep PUCT (their options are fixed; `tot` counts the backups from every parent).
//! * **Growth leaves.** A walk reaching an expanded other-seat node with an option outside its union ends there
//!   (`Sim::Leaf` on the expanded node): the caller evaluates this world's row (`leaf_acts`), `expand_leaf` grows
//!   the union and folds the priors into the means, the value is backed up along the path as any leaf's. A world
//!   whose options are all known walks on without a network row.
//! * **Reuse.** `advance_in` finds the new root by its real signature, else by its real observation (the live
//!   game's `encode(viewer)`: no hidden information in it); the map is then rebuilt from the nodes reachable from
//!   it, or `compact` (above 65k nodes) copies each reachable node once.
//!
//! The root (the viewer's node) is never merged into: it is on every walk's path. Deferred terminals' nodes are
//! not merged (a link to a fresh node, as the tree of before keys them).
use std::collections::{HashMap, HashSet};

use super::*;
use crate::lookahead::obs_hash_sparse;

/// A node's key in the observation map: (`obs_hash` of the viewer's features, `sig_hash` of the decider and -
/// for the viewer's own nodes - the options).
pub(super) type ObsKey = (u128, u64);

/// One child signature of an edge and the node it leads to.
#[derive(Clone, Copy, Debug)]
pub(super) struct Link {
    hash: u64,
    key_off: u32,
    key_len: u32,
    player: u8,
    acts_off: u32,
    nacts: u32,
    node: u32,
    next: u32,
}

/// Per stat (parallel to `Tree::stats`): the walks at its node in which the option was legal (other seats'
/// nodes), the priors folded into its mean (`Stat::p` = psum / pcnt there).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct XStat {
    avail: u32,
    pcnt: u32,
    psum: f64,
}

/// The merged tree's counters since `set_merge` (a diagnostic; `Tree::merge_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MergeStats {
    /// Links made (new child signatures).
    pub links: u64,
    /// Observations encoded (new signatures, and `advance_in`'s lookups).
    pub encodes: u64,
    /// New signatures that found an existing node by its observation (the merges).
    pub merged: u64,
    /// Walks given a fresh node because the observation's node was on their path.
    pub aliases: u64,
    /// Growth leaves (a world with an option outside an other seat's union).
    pub growths: u64,
}

/// The merged tree's state (`Tree::merge`).
#[derive(Clone, Debug)]
pub(super) struct Merge {
    links: Vec<Link>,
    /// Parallel to `Tree::stats`.
    xs: Vec<XStat>,
    /// Parallel to `Tree::nodes`: the node's key in the map (None: unmapped - an alias, a deferred terminal's
    /// node, the root of a fresh search).
    nobs: Vec<Option<ObsKey>>,
    map: HashMap<ObsKey, u32>,
    /// The last walk's growth leaf (NIL: none) and its world's options.
    pub(super) grow_node: u32,
    grow: Vec<u16>,
    count: MergeStats,
    // (encoding scratch: NUM_FEATURES zeros, the sparse pairs)
    dense: Vec<f32>,
    idx: Vec<u32>,
    val: Vec<f32>,
    /// The node whose state `idx` / `val` encode from the viewer's view: set by the walk's last `merged_child` when
    /// it encoded (a new signature), NIL otherwise (`Tree::leaf_obs`: the driver's row of the leaf's viewer).
    pub(super) enc_node: u32,
}

impl Merge {
    fn new(nodes: usize, stats: usize) -> Merge {
        Merge {
            links: Vec::new(),
            xs: vec![XStat::default(); stats],
            nobs: vec![None; nodes],
            map: HashMap::new(),
            grow_node: NIL,
            grow: Vec::new(),
            count: MergeStats::default(),
            dense: Vec::new(),
            idx: Vec::new(),
            val: Vec::new(),
            enc_node: NIL,
        }
    }

    /// A node was pushed (`Tree::push_node`): unmapped until the walk maps it.
    pub(super) fn pushed(&mut self) {
        self.nobs.push(None);
    }

    /// A node was expanded with these priors (`Tree::expand`): its side stats.
    pub(super) fn expanded(&mut self, priors: &[f32]) {
        for &p in priors {
            self.xs.push(XStat { avail: 0, pcnt: 1, psum: p as f64 });
        }
    }
}

fn on_path(path: &[(u32, u32)], k: u32) -> bool {
    path.iter().any(|&(n, _)| n == k)
}

impl<G: SearchGame> Tree<G> {
    /// Information-set merging for the search starting now (docs/search.md 3.16; off: the tree of before, bit
    /// for bit). A fresh tree only (right after `reset` / `new`, before its first simulation); `reset` turns it
    /// off. Meant for root-view leaf values (`driver::LeafView::Root`): an other seat's merged node is valued from
    /// the viewer's view, the same in every world of its observation (the driver refuses it otherwise).
    pub fn set_merge(&mut self, on: bool) -> Result<(), String> {
        if on == self.merge.is_some() {
            return Ok(());
        }
        if self.nodes.len() != 1 || self.expanded(self.root) {
            return Err("set_merge: a fresh tree only (right after reset, before its first simulation)".into());
        }
        self.merge = on.then(|| Box::new(Merge::new(self.nodes.len(), self.stats.len())));
        Ok(())
    }

    /// Whether the tree merges information sets (`set_merge`).
    pub fn merge_on(&self) -> bool {
        self.merge.is_some()
    }

    /// The merged tree's counters since `set_merge` (None when off).
    pub fn merge_stats(&self) -> Option<MergeStats> {
        self.merge.as_ref().map(|m| m.count)
    }

    /// The options of a leaf's network row: this world's for a merged tree's growth leaf (`Sim::Leaf` on an
    /// expanded node), else the node's (`acts`).
    pub fn leaf_acts(&self, node: u32) -> &[u16] {
        match &self.merge {
            Some(m) if m.grow_node == node => &m.grow,
            _ => self.acts(node),
        }
    }

    /// The viewer's features of the last walk's leaf `node` as the merged tree encoded them at the walk's last new
    /// signature (`encode_sparse(viewer)` of the leaf's state: its non-zero indexes and values), None when that walk
    /// did not encode its leaf (off, the root's own evaluation, a known link). The driver reuses them for the
    /// leaf's row of the viewer - the leaf's own row at a viewer's node, the root-view value-only row at another
    /// seat's - instead of encoding the same state again (docs/search.md 3.16). Valid until the next walk.
    pub fn leaf_obs(&self, node: u32) -> Option<(&[u32], &[f32])> {
        match &self.merge {
            Some(m) if m.enc_node == node && node != NIL => Some((&m.idx, &m.val)),
            _ => None,
        }
    }

    /// The links of (node, option i), newest first (a merged tree; empty if not expanded).
    pub(super) fn edge_links(&self, node: u32, i: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let (Some(m), so) = (self.merge.as_ref(), self.nodes[node as usize].stat_off) else { return out };
        if so == NIL {
            return out;
        }
        let mut l = self.stats[so as usize + i].kid;
        while l != NIL {
            out.push(l);
            l = m.links[l as usize].next;
        }
        out
    }

    pub(super) fn link_node(&self, l: u32) -> u32 {
        self.merge.as_ref().unwrap().links[l as usize].node
    }

    pub(super) fn link_key(&self, l: u32) -> &[u64] {
        let k = &self.merge.as_ref().unwrap().links[l as usize];
        &self.keys[k.key_off as usize..(k.key_off + k.key_len) as usize]
    }

    fn link_acts(&self, k: &Link) -> &[u16] {
        &self.acts[k.acts_off as usize..(k.acts_off + k.nacts) as usize]
    }

    /// The link of (node, option i) with this signature.
    fn find_link(&self, node: u32, i: usize, key: &[u64], player: usize, acts: &[u16], hash: u64) -> Option<u32> {
        let m = self.merge.as_ref()?;
        let so = self.nodes[node as usize].stat_off;
        if so == NIL {
            return None;
        }
        let mut l = self.stats[so as usize + i].kid;
        while l != NIL {
            let k = &m.links[l as usize];
            if k.hash == hash && k.player as usize == player && &self.keys[k.key_off as usize..(k.key_off + k.key_len) as usize] == key && self.link_acts(k) == acts {
                return Some(l);
            }
            l = k.next;
        }
        None
    }

    /// A new link at the head of stat `si`'s list.
    fn push_link(&mut self, si: usize, key: &[u64], player: usize, acts: &[u16], hash: u64, node: u32) {
        let key_off = self.keys.len() as u32;
        self.keys.extend_from_slice(key);
        let acts_off = self.acts.len() as u32;
        self.acts.extend_from_slice(acts);
        let next = self.stats[si].kid;
        let m = self.merge.as_mut().unwrap();
        let id = m.links.len() as u32;
        m.links.push(Link { hash, key_off, key_len: key.len() as u32, player: player as u8, acts_off, nacts: acts.len() as u32, node, next });
        m.count.links += 1;
        self.stats[si].kid = id;
    }

    /// The map key of the state `g` (decider `p`, options `acts`) seen by the viewer.
    fn obs_key(&mut self, g: &G, p: usize, acts: &[u16]) -> ObsKey {
        let viewer = self.viewer;
        let m = self.merge.as_mut().unwrap();
        if m.dense.len() != G::NUM_FEATURES {
            m.dense = vec![0.0; G::NUM_FEATURES];
        }
        m.idx.clear();
        m.val.clear();
        m.enc_node = NIL;
        g.encode_sparse(viewer, &mut m.dense, &mut m.idx, &mut m.val);
        m.count.encodes += 1;
        (obs_hash_sparse(&m.idx, &m.val, G::NUM_FEATURES), sig_hash(&[], p, if p == viewer { acts } else { &[] }))
    }

    /// The node the walk at (`parent`, option i) goes to after a transition to decision (p, `sc.legal`) with chance
    /// key `sc.key` (`g` = that state, `h` its signature hash): a known link's node, else the observation's node
    /// (or a new one), never a node on the walk's path. -> (node, whether it is a growth leaf).
    pub(super) fn merged_child(&mut self, parent: u32, i: usize, h: u64, g: &G, sc: &Scratch) -> (u32, bool) {
        let p = g.to_move().expect("merged_child: a decision");
        let si = self.nodes[parent as usize].stat_off as usize + i;
        self.merge.as_mut().unwrap().enc_node = NIL;
        let k = match self.find_link(parent, i, &sc.key, p, &sc.legal, h) {
            Some(l) => {
                let k = self.link_node(l);
                if on_path(&sc.path, k) {
                    let a = self.push_node(&sc.key, p, &sc.legal, h);
                    let m = self.merge.as_mut().unwrap();
                    m.links[l as usize].node = a;
                    m.count.aliases += 1;
                    a
                } else {
                    k
                }
            }
            None => {
                let ok = self.obs_key(g, p, &sc.legal);
                let k = match self.merge.as_ref().unwrap().map.get(&ok).copied() {
                    Some(k) if !on_path(&sc.path, k) => {
                        self.merge.as_mut().unwrap().count.merged += 1;
                        k
                    }
                    Some(_) => {
                        let a = self.push_node(&sc.key, p, &sc.legal, h);
                        self.merge.as_mut().unwrap().count.aliases += 1;
                        a
                    }
                    None => {
                        let k = self.push_node(&sc.key, p, &sc.legal, h);
                        let m = self.merge.as_mut().unwrap();
                        m.nobs[k as usize] = Some(ok);
                        m.map.insert(ok, k);
                        k
                    }
                };
                self.push_link(si, &sc.key, p, &sc.legal, h, k);
                // (the encoding is of `g`, the state at k in this world: the leaf's if the walk ends at k)
                self.merge.as_mut().unwrap().enc_node = k;
                k
            }
        };
        if !self.expanded(k) {
            // (a node is expanded by the walk that made it, so this is that walk's node, abandoned before its
            // evaluation: it takes this world's options - the viewer's own nodes are keyed by them anyway)
            if self.acts(k) != sc.legal.as_slice() {
                let off = self.acts.len() as u32;
                self.acts.extend_from_slice(&sc.legal);
                let nd = &mut self.nodes[k as usize];
                nd.acts_off = off;
                nd.nacts = sc.legal.len() as u32;
            }
            return (k, false);
        }
        if p != self.viewer && sc.legal.iter().any(|a| !self.acts(k).contains(a)) {
            let m = self.merge.as_mut().unwrap();
            m.grow_node = k;
            m.grow.clear();
            m.grow.extend_from_slice(&sc.legal);
            m.count.growths += 1;
            return (k, true);
        }
        (k, false)
    }

    /// A merged tree's deferred terminal under (node, option i) with chance key `key`: a known link's node, else a
    /// new unmapped node (deferred terminals are not merged).
    pub(super) fn deferred_child(&mut self, node: u32, i: usize, key: &[u64], h: u64) -> u32 {
        let v = self.viewer;
        if let Some(l) = self.find_link(node, i, key, v, &[], h) {
            return self.link_node(l);
        }
        let k = self.push_node(key, v, &[], h);
        let si = self.nodes[node as usize].stat_off as usize + i;
        self.push_link(si, key, v, &[], h, k);
        k
    }

    /// The subset-armed PUCT of another seat's merged node: among its options legal in this world (`legal`),
    /// argmax Q_a + c P'_a sqrt(max(avail_a, 1)) / (1 + N_a), P' = the mean priors renormalised over them
    /// (uniform if they sum to 0), Q_a = W_a / N_a or the node's network utility for its decider; ties -> the
    /// lowest index.
    pub(super) fn select_avail(&self, node: u32, c_puct: f64, legal: &[u16]) -> usize {
        let nd = &self.nodes[node as usize];
        let so = nd.stat_off as usize;
        let fpu = nd.vec[nd.player as usize];
        let acts = self.acts(node);
        let xs = &self.merge.as_ref().unwrap().xs;
        let (mut psum, mut navail) = (0.0, 0usize);
        for (j, a) in acts.iter().enumerate() {
            if legal.contains(a) {
                psum += self.stats[so + j].p as f64;
                navail += 1;
            }
        }
        let mut best_i = usize::MAX;
        let mut best = -1e30;
        for (j, a) in acts.iter().enumerate() {
            if !legal.contains(a) {
                continue;
            }
            let s = &self.stats[so + j];
            let q = if s.n > 0 { s.w / s.n as f64 } else { fpu };
            let pp = if psum > 0.0 { s.p as f64 / psum } else { 1.0 / navail as f64 };
            let u = q + c_puct * pp * (xs[so + j].avail.max(1) as f64).sqrt() / (1.0 + s.n as f64);
            if u > best {
                best_i = j;
                best = u;
            }
        }
        assert!(best_i != usize::MAX, "select_avail: none of the world's options {legal:?} is in the node's {acts:?}");
        best_i
    }

    /// A walk selected at another seat's merged node in a world with options `legal`: each of them was available.
    pub(super) fn mark_avail(&mut self, node: u32, legal: &[u16]) {
        let nd = &self.nodes[node as usize];
        let (so, off, na) = (nd.stat_off as usize, nd.acts_off as usize, nd.nacts as usize);
        let m = self.merge.as_mut().unwrap();
        for j in 0..na {
            if legal.contains(&self.acts[off + j]) {
                m.xs[so + j].avail += 1;
            }
        }
    }

    /// Expand a growth leaf (`expand` on the last walk's growth node): this world's options outside the union are
    /// appended (the node's options and stats move to the end of the flat arrays; the old copies are garbage until
    /// `compact`), and its priors join each option's mean.
    pub(super) fn grow(&mut self, node: u32, priors: &[f32]) {
        let m = self.merge.as_mut().unwrap();
        let legal = std::mem::take(&mut m.grow);
        m.grow_node = NIL;
        assert_eq!(priors.len(), legal.len(), "grow: priors for this world's options");
        let nd = self.nodes[node as usize].clone();
        let (so, na) = (nd.stat_off as usize, nd.nacts as usize);
        let old: Vec<u16> = self.acts(node).to_vec();
        let new: Vec<u16> = legal.iter().copied().filter(|a| !old.contains(a)).collect();
        if !new.is_empty() {
            let acts_off = self.acts.len() as u32;
            self.acts.extend_from_slice(&old);
            self.acts.extend_from_slice(&new);
            let stat_off = self.stats.len() as u32;
            let m = self.merge.as_mut().unwrap();
            for j in 0..na {
                self.stats.push(self.stats[so + j]);
                m.xs.push(m.xs[so + j]);
            }
            for _ in &new {
                self.stats.push(Stat { p: 0.0, p0: 0.0, n: 0, w: 0.0, kid: NIL, outs: NIL, nouts: 0 });
                m.xs.push(XStat::default());
            }
            let n2 = &mut self.nodes[node as usize];
            n2.acts_off = acts_off;
            n2.nacts = (na + new.len()) as u32;
            n2.stat_off = stat_off;
        }
        let so = self.nodes[node as usize].stat_off as usize;
        for (a, &p) in legal.iter().zip(priors) {
            let j = so + self.acts(node).iter().position(|x| x == a).unwrap();
            let m = self.merge.as_mut().unwrap();
            let x = &mut m.xs[j];
            x.psum += p as f64;
            x.pcnt += 1;
            let mean = (x.psum / x.pcnt as f64) as f32;
            self.stats[j].p = mean;
            self.stats[j].p0 = mean;
        }
    }

    /// Per option of a merged node: the walks at it in which the option was legal (other seats' nodes; empty if
    /// not expanded or not merged).
    pub fn avail(&self, node: u32) -> Vec<u32> {
        let (Some(m), nd) = (self.merge.as_ref(), &self.nodes[node as usize]) else { return Vec::new() };
        if nd.stat_off == NIL {
            return Vec::new();
        }
        m.xs[nd.stat_off as usize..(nd.stat_off + nd.nacts) as usize].iter().map(|x| x.avail).collect()
    }

    /// The links into each node reachable from the root (a merged tree's diagnostic: a node with two or more was
    /// reached by different signatures - other worlds, other parents). Empty when not merged.
    pub fn in_links(&self) -> HashMap<u32, u32> {
        let mut out = HashMap::new();
        if self.merge.is_none() {
            return out;
        }
        for id in self.reachable() {
            let nd = &self.nodes[id as usize];
            if nd.stat_off == NIL {
                continue;
            }
            for i in 0..nd.nacts as usize {
                for l in self.edge_links(id, i) {
                    *out.entry(self.link_node(l)).or_insert(0) += 1;
                }
            }
        }
        out
    }

    /// The nodes reachable from the root through links, each once, breadth first.
    pub(super) fn reachable(&self) -> Vec<u32> {
        let m = self.merge.as_ref().unwrap();
        let mut seen = HashSet::new();
        seen.insert(self.root);
        let mut order = vec![self.root];
        let mut k = 0;
        while k < order.len() {
            let id = order[k];
            k += 1;
            let nd = &self.nodes[id as usize];
            if nd.stat_off == NIL {
                continue;
            }
            for i in 0..nd.nacts {
                let mut l = self.stats[(nd.stat_off + i) as usize].kid;
                while l != NIL {
                    let c = m.links[l as usize].node;
                    if seen.insert(c) {
                        order.push(c);
                    }
                    l = m.links[l as usize].next;
                }
            }
        }
        order
    }

    /// The observation map of the nodes reachable from the root (after the root moved).
    fn rebuild_map(&mut self) {
        let order = self.reachable();
        let m = self.merge.as_mut().unwrap();
        m.map.clear();
        for &n in &order {
            if let Some(k) = m.nobs[n as usize] {
                m.map.insert(k, n);
            }
        }
    }

    /// `advance` with the live game after the move (`live`: its next decision is (`player`, `legal`)). A merged tree
    /// (docs/search.md 3.16) finds the new root by the real signature, else by the real observation (`live`'s
    /// `encode(viewer)`), only for the same viewer, then rebuilds its map (or compacts); otherwise `advance`.
    pub fn advance_in(&mut self, a: u16, key: &[u64], player: usize, legal: &[u16], live: &G) -> bool {
        if self.merge.is_none() {
            return self.advance(a, key, player, legal);
        }
        let root = self.root;
        if !self.expanded(root) || player != self.viewer {
            return false;
        }
        let Some(i) = self.acts(root).iter().position(|&x| x == a) else { return false };
        let h = sig_hash(key, player, legal);
        let k = match self.find_link(root, i, key, player, legal, h) {
            Some(l) => Some(self.link_node(l)),
            None => {
                let ok = self.obs_key(live, player, legal);
                self.merge.as_ref().unwrap().map.get(&ok).copied()
            }
        };
        let Some(k) = k.filter(|&k| k != root && self.player(k) == player && self.acts(k) == legal) else { return false };
        self.root = k;
        self.set_root_min(0);
        self.belief = None;
        self.belief_mix = 0.0;
        self.gumbel = None;
        let m = self.merge.as_mut().unwrap();
        m.grow_node = NIL;
        m.enc_node = NIL;
        if self.nodes.len() > COMPACT_AT {
            self.compact();
        } else {
            self.rebuild_map();
        }
        true
    }

    /// `compact` of a merged tree: each node reachable from the root copied once (breadth first, the root 0), its
    /// links (in order), outs and side stats; the map rebuilt.
    pub(super) fn compact_merged(&mut self) {
        let order = self.reachable();
        let new_id: HashMap<u32, u32> = order.iter().enumerate().map(|(n, &o)| (o, n as u32)).collect();
        let om = self.merge.take().unwrap();
        let mut t: Tree<G> = Tree {
            viewer: self.viewer,
            redet: self.redet,
            n: self.n,
            team: self.team,
            root: 0,
            base: self.base.take(),
            root_min: self.root_min,
            forced: std::mem::take(&mut self.forced),
            forced_win: self.forced_win,
            forced_n: self.forced_n,
            walk_forced: self.walk_forced,
            belief: self.belief.take(),
            belief_mix: self.belief_mix,
            gumbel: self.gumbel.take(),
            ..Default::default()
        };
        let mut nm = Merge::new(0, 0);
        nm.count = om.count;
        for &old in &order {
            let nd = &self.nodes[old as usize];
            let mut n2 = nd.clone();
            n2.acts_off = t.acts.len() as u32;
            t.acts.extend_from_slice(&self.acts[nd.acts_off as usize..(nd.acts_off + nd.nacts) as usize]);
            n2.key_off = t.keys.len() as u32;
            t.keys.extend_from_slice(&self.keys[nd.key_off as usize..(nd.key_off + nd.key_len) as usize]);
            n2.next = NIL;
            if nd.stat_off != NIL {
                n2.stat_off = t.stats.len() as u32;
                for i in 0..nd.nacts {
                    let si = (nd.stat_off + i) as usize;
                    let mut s = self.stats[si];
                    let mut links = Vec::new();
                    let mut l = s.kid;
                    while l != NIL {
                        links.push(om.links[l as usize]);
                        l = om.links[l as usize].next;
                    }
                    let mut next = NIL;
                    for lk in links.iter().rev() {
                        let key_off = t.keys.len() as u32;
                        t.keys.extend_from_slice(&self.keys[lk.key_off as usize..(lk.key_off + lk.key_len) as usize]);
                        let acts_off = t.acts.len() as u32;
                        t.acts.extend_from_slice(self.link_acts(lk));
                        let id = nm.links.len() as u32;
                        nm.links.push(Link { key_off, acts_off, node: new_id[&lk.node], next, ..*lk });
                        next = id;
                    }
                    s.kid = next;
                    let mut recs = Vec::new();
                    let mut o = s.outs;
                    while o != NIL {
                        recs.push(self.outs[o as usize]);
                        o = self.outs[o as usize].next;
                    }
                    let mut next = NIL;
                    for r in recs.iter().rev() {
                        let key_off = t.keys.len() as u32;
                        t.keys.extend_from_slice(&self.keys[r.key_off as usize..(r.key_off + r.key_len) as usize]);
                        let id = t.outs.len() as u32;
                        t.outs.push(Out { key_off, key_len: r.key_len, next });
                        next = id;
                    }
                    s.outs = next;
                    t.stats.push(s);
                    nm.xs.push(om.xs[si]);
                }
            }
            t.nodes.push(n2);
            nm.nobs.push(om.nobs[old as usize]);
        }
        for (n, k) in nm.nobs.iter().enumerate() {
            if let Some(k) = k {
                nm.map.insert(*k, n as u32);
            }
        }
        t.merge = Some(Box::new(nm));
        *self = t;
    }
}

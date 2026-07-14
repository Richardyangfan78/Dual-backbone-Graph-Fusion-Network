"""Generate 4 paper figures from v7 CGCNN-MT: SHAP, BG scatter, GT CM, EH CM."""
import sys, os, csv, random
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_absolute_error
import warnings; warnings.filterwarnings('ignore')

PROJECT = '/path/to/Dual-backbone-Graph-Fusion-Network'
sys.path.insert(0, os.path.join(PROJECT, 'multitask'))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, 'cgcnn'))
os.chdir(PROJECT)

from model_mt_v7 import CrystalGraphConvNetMTV7
from data_mt_v4 import stratified_kfold_v4, _map_gaptype

DATA_DIR  = os.path.join(PROJECT, 'Data/multitask')
CACHE_DIR = os.path.join(DATA_DIR, 'cached_graphs')
CKPT_DIR  = os.path.join(PROJECT, 'checkpoints/multitask_v7')
OUT_DIR   = os.path.join(PROJECT, 'figures_paper')
os.makedirs(OUT_DIR, exist_ok=True)

STABLE_THRESH = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def eh_label(v): return 0 if float(v) < STABLE_THRESH else 1

class CachedDS(torch.utils.data.Dataset):
    def __init__(self, cache_dir, id_prop_data):
        self.cache_dir = cache_dir; self.data = id_prop_data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        r = self.data[idx]
        d = torch.load(os.path.join(self.cache_dir, f'{r[0]}.pt'), weights_only=True)
        return (d['atom_fea'], d['nbr_fea'], d['nbr_fea_idx']), {
            'bandgap':  d['bandgap'],
            'gaptype':  torch.LongTensor([_map_gaptype(r[2], True)]),
            'eh_label': torch.LongTensor([eh_label(r[3])]),
        }, r[0]

def collate(batch):
    af,nf,ni,ci,bg,gt,eh = [],[],[],[],[],[],[]
    base=0
    for (a,n,ni_),t,_ in batch:
        s=a.shape[0]; af.append(a); nf.append(n)
        ni.append(ni_+base); ci.append(torch.arange(s)+base)
        bg.append(t['bandgap']); gt.append(t['gaptype']); eh.append(t['eh_label'])
        base+=s
    return (torch.cat(af),torch.cat(nf),torch.cat(ni),ci), \
           {'bandgap':torch.stack(bg),'gaptype':torch.stack(gt),'eh_label':torch.stack(eh)}, []

def apply_rules(bg, gt_lp):
    p=gt_lp.argmax(1).clone(); p[bg.squeeze(-1)<0.05]=1; return p

class Norm:
    def load_state_dict(self,d): self.mean=d['mean']; self.std=d['std']
    def denorm(self,x): return x*self.std+self.mean

# ── Load data & folds ─────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR,'id_prop.csv')) as f:
    ipd = list(csv.reader(f))
random.seed(123); random.shuffle(ipd)

ds = CachedDS(CACHE_DIR, ipd)
samp = torch.load(os.path.join(CACHE_DIR,f'{ipd[0][0]}.pt'),weights_only=True)
oaf = samp["atom_fea"].shape[1]; nbf = samp["nbr_fea"].shape[2]
print(f'atom_fea={oaf}, nbr_fea={nbf}')

# EXACT same call as training script (random_seed default=123, merge=True)
folds = stratified_kfold_v4(ipd, n_folds=3, merge_metal_indirect=True)

# ── Feature groups ────────────────────────────────────────────────────────────
FG = {
    'Group (1-hot)':              list(range(0,18)),
    'Period (1-hot)':             list(range(18,25)),
    'Electronegativity bin':      list(range(25,35)),
    'Valence electrons (s/p/d/f)':list(range(35,42)),
    'Oxidation states':           list(range(42,55)),
    'Ionisation energy bin':      list(range(55,65)),
    'Atomic radius bin':          list(range(65,75)),
    'Coordination number':        list(range(75,84)),
    'Other binary features':      list(range(84,92)),
    'Electronegativity (cont.)':  [92],
    'Spin-orbit coupling ξ':      [93],
    'Ionic radius (cont.)':       [94],
}
fg_names = list(FG.keys())
fg_idx   = list(FG.values())

# ── Per-fold inference ────────────────────────────────────────────────────────
all_bg_t,all_bg_p = [],[]
all_gt_t,all_gt_p = [],[]
all_eh_t,all_eh_p = [],[]
all_imp = []

for fi in range(3):
    ckpt = os.path.join(CKPT_DIR,f'fold_{fi}_seed_42','best_composite.pth.tar')
    if not os.path.isfile(ckpt): print(f'MISSING {ckpt}'); continue
    print(f'\nFold {fi}')

    test_idx = folds[fi]
    m = CrystalGraphConvNetMTV7(oaf,nbf,atom_fea_len=128,n_conv=6,h_fea_len=256,
                                n_gap_classes=2,n_eh_classes=2,dropout=0.0,n_attn_heads=8).to(device)
    nb = Norm()
    c = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(c['state_dict']); nb.load_state_dict(c['normalizer_bg'])
    m.eval()

    ldr = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds,test_idx), batch_size=32,
        shuffle=False, collate_fn=collate, num_workers=0)

    fold_imp = np.zeros(oaf); nb_batches=0
    for (af,nf,ni,ci), tgt, _ in ldr:
        # --- gradient attribution ---
        af_g = af.to(device).requires_grad_(True)
        b,g,e = m(af_g, nf.to(device), ni.to(device), ci)
        (b.sum()+g.sum()+e.sum()).backward()
        fold_imp += np.abs(af_g.grad.detach().cpu().numpy() * af.numpy()).mean(0)
        nb_batches+=1; m.zero_grad()
        # --- preds ---
        with torch.no_grad():
            pb,pg,pe = m(af.to(device),nf.to(device),ni.to(device),ci)
        all_bg_p.extend(nb.denorm(pb).cpu().numpy().flatten())
        all_bg_t.extend(tgt['bandgap'].numpy().flatten())
        all_gt_p.extend(apply_rules(pb.cpu(),pg.cpu()).numpy())
        all_gt_t.extend(tgt['gaptype'].squeeze(-1).numpy())
        all_eh_p.extend(pe.cpu().argmax(1).numpy())
        all_eh_t.extend(tgt['eh_label'].squeeze(-1).numpy())
    all_imp.append(fold_imp/max(nb_batches,1))
    print(f'  {len(test_idx)} samples done')

importance = np.mean(all_imp,0)
grp_imp = np.array([importance[idx].sum() for idx in fg_idx])
order = np.argsort(grp_imp)  # ascending → for hbar, goes bottom→top

bg_t=np.array(all_bg_t); bg_p=np.array(all_bg_p)
gt_t=np.array(all_gt_t,int); gt_p=np.array(all_gt_p,int)
eh_t=np.array(all_eh_t,int); eh_p=np.array(all_eh_p,int)

mae=mean_absolute_error(bg_t,bg_p); r=np.corrcoef(bg_t,bg_p)[0,1]
gt_acc=accuracy_score(gt_t,gt_p); gt_f1=f1_score(gt_t,gt_p,average='binary',zero_division=0)
eh_acc=accuracy_score(eh_t,eh_p); eh_f1=f1_score(eh_t,eh_p,average='binary',zero_division=0)

print(f'\nBG  MAE={mae:.4f} r={r:.4f}  GT Acc={gt_acc:.4f} F1={gt_f1:.4f}  EH Acc={eh_acc:.4f} F1={eh_f1:.4f}')

# ── STYLE ──────────────────────────────────────────────────────────────────────
BLUE='#2C6FAC'; LBLUE='#BDD5F0'; GREEN='#2E8B57'; LGREEN='#BCDEC6'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})

# ════ Fig 1: Feature importance ═══════════════════════════════════════════════
fig1,ax1 = plt.subplots(figsize=(7.5,4.8))
cols = [BLUE if grp_imp[i]>=np.sort(grp_imp)[-3] else LBLUE for i in order]
ax1.barh([fg_names[i] for i in order], grp_imp[order], color=cols, edgecolor='white', height=0.65)
ax1.set_xlabel('Mean |Gradient × Input| (summed per feature group)')
ax1.set_title('CGCNN-MT: Atom Feature Importance\n(Gradient×Input attribution, 3-fold test sets, v7)')
ax1.xaxis.grid(True,alpha=0.25,ls='--'); ax1.set_axisbelow(True)
ax1.tick_params(axis='y',labelsize=9.5)
ax1.legend(handles=[Patch(color=BLUE,label='Top-3 groups'),Patch(color=LBLUE,label='Other groups')],
           fontsize=9,loc='lower right')
plt.tight_layout(); fig1.savefig(os.path.join(OUT_DIR,'fig_shap_importance.png'),dpi=180,bbox_inches='tight')
print('Saved fig_shap_importance.png')

# ════ Fig 2: BG scatter ════════════════════════════════════════════════════════
mx=max(bg_t.max(),bg_p.max())*1.05
fig2,ax2=plt.subplots(figsize=(4.8,4.8))
ax2.scatter(bg_t,bg_p,s=16,alpha=0.5,color=BLUE,edgecolors='none',rasterized=True)
ax2.plot([0,mx],[0,mx],'k--',lw=1.2)
ax2.set_xlim(0,mx); ax2.set_ylim(0,mx)
ax2.set_xlabel('DFT-PBE Bandgap (eV)'); ax2.set_ylabel('Predicted Bandgap (eV)')
ax2.set_title(f'Bandgap Regression (v7)\n3-fold CV, best-composite checkpoints')
ax2.text(0.05,0.93,f'MAE = {mae:.3f} eV\nr = {r:.3f}\nn = {len(bg_t)}',
         transform=ax2.transAxes,fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.85))
plt.tight_layout(); fig2.savefig(os.path.join(OUT_DIR,'fig_bandgap_scatter.png'),dpi=180,bbox_inches='tight')
print('Saved fig_bandgap_scatter.png')

# ════ Fig 3: Gap type CM ═══════════════════════════════════════════════════════
cm_gt=confusion_matrix(gt_t,gt_p); lgt=['Direct','Indirect/Metal']
fig3,ax3=plt.subplots(figsize=(4.2,3.8))
im3=ax3.imshow(cm_gt,cmap='Blues',vmin=0)
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(lgt,fontsize=10); ax3.set_yticklabels(lgt,fontsize=10)
ax3.set_xlabel('Predicted',labelpad=5); ax3.set_ylabel('True',labelpad=5)
ax3.set_title(f'Gap Type Classification (v7)\nAcc = {gt_acc:.3f} | F1 = {gt_f1:.3f}')
cmn=cm_gt/cm_gt.sum(axis=1,keepdims=True)
for i in range(2):
    for j in range(2):
        ax3.text(j,i,f'{cm_gt[i,j]}\n({cmn[i,j]:.1%})',ha='center',va='center',
                 fontsize=11,fontweight='bold',color='white' if cmn[i,j]>0.5 else 'black')
plt.colorbar(im3,ax=ax3,fraction=0.046,pad=0.04)
plt.tight_layout(); fig3.savefig(os.path.join(OUT_DIR,'fig_gaptype_cm.png'),dpi=180,bbox_inches='tight')
print('Saved fig_gaptype_cm.png')

# ════ Fig 4: EH stability CM ════════════════════════════════════════════════════
cm_eh=confusion_matrix(eh_t,eh_p); leh=['Stable\n(EH<0.01)','Unstable\n(EH≥0.01)']
fig4,ax4=plt.subplots(figsize=(4.2,3.8))
im4=ax4.imshow(cm_eh,cmap='Greens',vmin=0)
ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
ax4.set_xticklabels(leh,fontsize=10); ax4.set_yticklabels(leh,fontsize=10)
ax4.set_xlabel('Predicted',labelpad=5); ax4.set_ylabel('True',labelpad=5)
ax4.set_title(f'EH Stability Classification (v7)\nAcc = {eh_acc:.3f} | F1 = {eh_f1:.3f}')
cmn4=cm_eh/cm_eh.sum(axis=1,keepdims=True)
for i in range(2):
    for j in range(2):
        ax4.text(j,i,f'{cm_eh[i,j]}\n({cmn4[i,j]:.1%})',ha='center',va='center',
                 fontsize=11,fontweight='bold',color='white' if cmn4[i,j]>0.5 else 'black')
plt.colorbar(im4,ax=ax4,fraction=0.046,pad=0.04)
plt.tight_layout(); fig4.savefig(os.path.join(OUT_DIR,'fig_eh_cm.png'),dpi=180,bbox_inches='tight')
print('Saved fig_eh_cm.png')

print(f'\nAll 4 figures saved to: {OUT_DIR}')

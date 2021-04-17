import os


def dfs(cwd, upd_fn):
    file_names = os.listdir(cwd)
    for name in file_names:
        path = os.path.join(cwd, name)
        if os.path.isdir(path):
            dfs(path, upd_fn)
        else:
            # if name == 'cfg.yaml' or name.endswith('.sh'):
            if name.endswith('.yaml'):
                upd_fn(path)


if __name__ == '__main__':
    def upd(path):
        with open(path, 'r', encoding='utf-8') as fp:
            ctt = fp.read()
        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(
                ctt.replace(
"""
checkpoints: [
   /mnt/lustre/tiankeyu/htl_ckpt/GS_MTL_LV1_10_and_LV2eve_R50.pth.tar,
   /mnt/lustre/tiankeyu/htl_ckpt/GS_MTL_LV1_10_and_LV2eve_4K_R50.pth.tar,
   /mnt/lustre/tiankeyu/htl_ckpt/DY_MTL_LV1_10_R50_convertBB.pth.tar,
   /mnt/lustre/tiankeyu/htl_ckpt/xueshuClip.pth.tar,
]   # should be less than num_gpus
""",
"""
checkpoints: [
   /mnt/lustre/tiankeyu/htl_ckpt/GS_MTL_LV1_10_and_LV2eve_R50.pth.tar,
   /mnt/lustre/tiankeyu/htl_ckpt/GS_MTL_LV1_10_and_LV2eve_4K_R50.pth.tar,
   /mnt/lustre/tiankeyu/htl_ckpt/DY_MTL_LV1_10_R50_convertBB.pth.tar,
   /mnt/lustre/tiankeyu/htl_ckpt/xueshuClip.pth.tar,
]   # len(checkpoints) should be less than num_gpus
""",
                )
            )
    
    dfs(os.getcwd(), upd)
    


# 1. matplotlib.patchesモジュールの読み込み
import matplotlib.pyplot as plt
from matplotlib import patches

# 2.	Axesオブジェクト生成
fig, ax = plt.subplots(figsize=(4,4))
 
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, -1, 0, 1, 2])
ax.grid()
 
# 3. 図形オブジェクト生成
c = patches.Circle( xy=(0,0), radius=1) # 円のオブジェクト
r = patches.Rectangle( xy=(1,1) , width=1, height=1) # 四角形のオブジェクト
 
# 4. Axesに図形オブジェクト追加・表示
ax.add_patch(c)
ax.add_patch(r)
plt.show()

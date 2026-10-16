"""Render every suggested hand-identity correction before applying it."""
import json
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/episode_000011'
audit=json.loads((BASE/'hand_sequence_hybrid/identity_audit.json').read_text())
tiles=[]
for change in audit['changes']:
    frame=change['frame_index']
    im=Image.open(BASE/f'calibrated_sequence/frames/{frame:06d}.jpg').convert('RGB')
    draw=ImageDraw.Draw(im)
    for box,raw,side in zip(change['boxes'],change['raw_is_right'],change['suggested_is_right']):
        draw.rectangle(box,outline='lime' if side else 'magenta',width=4)
        text=f'{"R" if raw else "L"}->{"R" if side else "L"}'
        draw.text((max(0,min(960,box[0])),max(0,box[1]-20)),text,fill='white',stroke_width=2,stroke_fill='black')
    im=im.resize((384,288));draw=ImageDraw.Draw(im)
    draw.rectangle((0,0,130,23),fill='black');draw.text((5,5),f'frame {frame}',fill='white')
    tiles.append(im)
sheet=Image.new('RGB',(384*3,288*((len(tiles)+2)//3)))
for i,im in enumerate(tiles):sheet.paste(im,((i%3)*384,(i//3)*288))
path=BASE/'hand_sequence_hybrid/identity_review_repaired_candidates.jpg'
sheet.save(path,quality=94)
print(path)

from pathlib import Path
root=Path(__file__).resolve().parent
html=(root/'index.html').read_text(encoding='utf-8').replace('/*__CONTROL_PROTOCOL__*/',(root/'control_protocol.js').read_text(encoding='utf-8'))
redirect='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/"><a href="/">返回关节控制</a>'
out=root.parent/'.vendor/wifi-debug/RoArm-M3_example/m3_web_page.h'
out.write_text('static const char index_html[] PROGMEM = R"ROARMUI('+html+')ROARMUI";\n'+
 '\n'.join(f'const char {name}[] PROGMEM = R"ROARMUI({redirect})ROARMUI";' for name in ['horizontal_drag_html','vertical_drag_html']),encoding='utf-8')
print(out)
(root/'index.preview.html').write_text(html,encoding='utf-8')

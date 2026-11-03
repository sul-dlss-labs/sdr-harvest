import sys
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from PIL import Image

NS = "http://www.loc.gov/standards/alto/ns-v4#"
ET.register_namespace("", NS)
q = lambda t: f"{{{NS}}}{t}"

GRAPHIC = {"Image", "Figure", "Diagram", "Chemical-Block"}

def convert(html, image_path, scale=1000, size=None):
    """size=(width, height) overrides reading dimensions from image_path."""
    W, H = size or Image.open(image_path).size
    alto = ET.Element(q("alto"))
    desc = ET.SubElement(alto, q("Description"))
    ET.SubElement(desc, q("MeasurementUnit")).text = "pixel"
    src = ET.SubElement(desc, q("sourceImageInformation"))
    ET.SubElement(src, q("fileName")).text = image_path
    page = ET.SubElement(ET.SubElement(alto, q("Layout")), q("Page"),
                         ID="P1", WIDTH=str(W), HEIGHT=str(H), PHYSICAL_IMG_NR="1")
    ps = ET.SubElement(page, q("PrintSpace"), HPOS="0", VPOS="0",
                       WIDTH=str(W), HEIGHT=str(H))
    n = 0
    for div in BeautifulSoup(html, "html.parser").find_all("div", recursive=False):
        label = div.get("data-label") or "block"
        if label == "Blank-Page":
            continue
        try:
            x0, y0, x1, y1 = map(int, div["data-bbox"].split())
        except Exception:
            continue
        hpos, vpos = int(x0 * W / scale), int(y0 * H / scale)
        w = max(1, int(x1 * W / scale) - hpos)
        h = max(1, int(y1 * H / scale) - vpos)
        geo = dict(HPOS=str(hpos), VPOS=str(vpos), WIDTH=str(w), HEIGHT=str(h))
        n += 1
        if label in GRAPHIC:
            ET.SubElement(ps, q("Illustration"), ID=f"B{n}", TYPE=label, **geo)
            continue
        tb = ET.SubElement(ps, q("TextBlock"), ID=f"B{n}", **geo)
        items = ([li.get_text(" ", strip=True) for li in div.find_all("li")]
                 or [p.get_text(" ", strip=True) for p in div.find_all("p")]
                 or [div.get_text(" ", strip=True)])
        items = [t for t in items if t]
        lh = max(1, h // max(1, len(items)))
        for i, text in enumerate(items):
            y = vpos + i * lh
            tl = ET.SubElement(tb, q("TextLine"), ID=f"B{n}_L{i+1}",
                               HPOS=str(hpos), VPOS=str(y), WIDTH=str(w), HEIGHT=str(lh))
            words = text.split()
            total = sum(len(x) for x in words) or 1
            x = hpos
            for j, word in enumerate(words):
                ww = max(1, int(w * len(word) / total))  # estimated width
                ET.SubElement(tl, q("String"), ID=f"B{n}_L{i+1}_W{j+1}", CONTENT=word,
                              HPOS=str(x), VPOS=str(y), WIDTH=str(ww), HEIGHT=str(lh))
                x += ww
                if j < len(words) - 1:
                    ET.SubElement(tl, q("SP"))
    ET.indent(alto)
    return ET.tostring(alto, encoding="unicode", xml_declaration=True)

if __name__ == "__main__":
    print(convert(open(sys.argv[1]).read(), sys.argv[2]))

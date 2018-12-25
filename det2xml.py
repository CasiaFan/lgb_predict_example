from lxml import etree, objectify

def instance2xml_base(filename, img_h, img_w):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/example'),
        E.filename(filename),
        E.source(
            E.database('MS COCO 2014'),
            E.annotation('MS COCO 2014'),
            E.image('Personal'),
        ),
        E.size(
            E.width(img_w),
            E.height(img_h),
            E.depth(3)
        ),
        E.segmented(0),
    )
    return anno_tree


def instance2xml_bbox(bbox, category_id):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    xmin, ymin, xmax, ymax = bbox
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.object(
        E.name(category_id),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax)
        ),
        E.difficult(0)
    )
    return anno_tree


def write_xml(anno, save_file):
    """
    anno: dict with following input items:
          image_name, img_h, img_w, category_id_list (list of category id correspond to bbox_list),
          bbox_list (list of bbox with [xmin, ymin, xmax, ymax])
    """
    anno_tree = instance2xml_base(anno["filename"],  anno["img_h"], anno["img_w"])
    cate_list = anno["category_id"]
    for i, bbox in enumerate(anno["bbox_list"]):
        anno_tree.append(instance2xml_bbox(bbox, cate_list[i]))
    etree.ElementTree(anno_tree).write(save_file, pretty_print=True)
    print("Formating instance xml file {} done!".format(anno["filename"]))


def test():
    anno = {"filename": "/usr/x/1/jpg",
            "img_h": 40,
            "img_w": 50,
            "category_id": [1],
            "bbox_list": [[1,2,3,4]]}
    save_file = "test.xml"
    write_xml(anno, save_file)


if __name__ == "__main__":
    test()
import fitz  # 需要安装：pip install PyMuPDF


def extract_foxit_annotations(pdf_path):
    doc = fitz.open(pdf_path)
    found_annotations = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        # 获取所有注释并转为列表
        annots = list(page.annots())

        if annots:
            found_annotations = True
            print(f"第{page_num + 1}页有{len(annots)}个注释:")

            for i, annot in enumerate(annots):
                print(f"注释 {i + 1}:")
                print(f"类型: {annot.type[1] if hasattr(annot, 'type') and len(annot.type) > 1 else '未知'}")
                print(f"内容: {annot.info.get('content', '无内容')}")
                print("-" * 40)

    if not found_annotations:
        print("未检测到注释")

    doc.close()


# 使用方法
extract_foxit_annotations('/home/ubuntu/Desktop/low_light_SOD (4).pdf')

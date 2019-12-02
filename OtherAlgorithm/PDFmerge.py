import os
import os.path
from PyPDF2 import PdfFileReader,PdfFileWriter
import time
t = time.time()


def getFileName(file_path):
    file_list = []
    for root,dirs,files in os.walk(file_path):
        for filespath in files:
            file_list.append(os.path.join(root,filespath))
    return file_list



def MergePDF(filepath,outfile):
    output = PdfFileWriter()
    outputPages = 0
    pdf_fileName = getFileName(filepath)
    pdf_fileName = sorted(pdf_fileName)
    for each in pdf_fileName:
        input = PdfFileReader(open(each, "rb"), strict = False)
        if input.isEncrypted == True:
            input.decrypt("map")

        # 获得源pdf文件中页面总数
        pageCount = input.getNumPages()
        outputPages += pageCount
        print(pageCount)

        # 分别将page添加到输出output中
        for iPage in range(0, pageCount):
            output.addPage(input.getPage(iPage))


    print("All Pages Number:" + str(outputPages))
    # 最后写pdf文件
    outputStream = open(filepath + outfile,"wb")
    output.write(outputStream)
    outputStream.close()
    print("finished")



if __name__ == '__main__':
    file_dir = r'/Users/pengbohao/Desktop/filee'
    out=u"application.pdf"
    MergePDF(file_dir,out)

    print('time cost: ', time.time() - t)


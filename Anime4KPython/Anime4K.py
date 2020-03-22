import cv2


class Anime4K(object):
    def __init__(
        self, passes=2, strengthColor=1 / 3, strengthGradient=1, fastMode=False,
    ):
        # Passes for processing
        self.ps = passes
        # the range of strengthColor from 0 to 1, greater for thinner linework
        self.sc = strengthColor
        # the range of strengthGradient from 0 to 1, greater for sharper
        self.sg = strengthGradient
        # Faster but may be low quality
        self.fm = fastMode

    def loadImage(self, path="./Anime4K/pic/p1.png"):
        self.srcFile = path
        self.orgImg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.dstImg = cv2.resize(
            self.orgImg, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        self.H = self.dstImg.shape[0]
        self.W = self.dstImg.shape[1]

    # Process anime4k
    def process(self):
        for i in range(self.ps):
            self.getGray()
            self.pushColor()
            self.getGradient()
            self.pushGradient()

    # getGray compute the grayscale of the image and store it to the Alpha channel
    def getGray(self):
        B, G, R, A = 0, 1, 2, 3

        def callBack(i, j, pixel):
            pixel[A] = 0.299 * pixel[R] + 0.587 * pixel[G] + 0.114 * pixel[B]
            return pixel

        self.changeEachPixel(self.dstImg, callBack)

    # pushColor will make the linework of the image thinner guided by the grayscale in Alpha channel
    def pushColor(self):
        B, G, R, A = 0, 1, 2, 3

        def getLightest(mc, a, b, c):
            mc[R] = mc[R] * (1 - self.sc) + (a[R] / 3 + b[R] / 3 + c[R] / 3) * self.sc
            mc[G] = mc[G] * (1 - self.sc) + (a[G] / 3 + b[G] / 3 + c[G] / 3) * self.sc
            mc[B] = mc[B] * (1 - self.sc) + (a[B] / 3 + b[B] / 3 + c[B] / 3) * self.sc
            mc[A] = mc[A] * (1 - self.sc) + (a[A] / 3 + b[A] / 3 + c[A] / 3) * self.sc

        def callBack(i, j, pixel):
            iN, iP, jN, jP = -1, 1, -1, 1
            if i == 0:
                iN = 0
            elif i == self.H - 1:
                iP = 0
            if j == 0:
                jN = 0
            elif j == self.W - 1:
                jP = 0

            tl, tc, tr = (
                self.dstImg[i + iN, j + jN],
                self.dstImg[i + iN, j],
                self.dstImg[i + iN, j + jP],
            )
            ml, mc, mr = (
                self.dstImg[i, j + jN],
                pixel,
                self.dstImg[i, j + jP],
            )
            bl, bc, br = (
                self.dstImg[i + iP, j + jN],
                self.dstImg[i + iP, j],
                self.dstImg[i + iP, j + jP],
            )

            # top and bottom
            maxD = max(bl[A], bc[A], br[A])
            minL = min(tl[A], tc[A], tr[A])
            if minL > mc[A] and mc[A] > maxD:
                getLightest(mc, tl, tc, tr)
            else:
                maxD = max(tl[A], tc[A], tr[A])
                minL = min(bl[A], bc[A], br[A])
                if minL > mc[A] and mc[A] > maxD:
                    getLightest(mc, bl, bc, br)

            # subdiagonal
            maxD = max(ml[A], mc[A], bc[A])
            minL = min(tc[A], tr[A], mr[A])
            if minL > maxD:
                getLightest(mc, tc, tr, mr)
            else:
                maxD = max(tc[A], mc[A], mr[A])
                minL = min(ml[A], bl[A], bc[A])
                if minL > maxD:
                    getLightest(mc, ml, bl, bc)

            # left and right
            maxD = max(tl[A], ml[A], bl[A])
            minL = min(tr[A], mr[A], br[A])
            if minL > mc[A] and mc[A] > maxD:
                getLightest(mc, tr, mr, br)
            else:
                maxD = max(tr[A], mr[A], br[A])
                minL = min(tl[A], ml[A], bl[A])
                if minL > mc[A] and mc[A] > maxD:
                    getLightest(mc, tl, ml, bl)

            # diagonal
            maxD = max(tc[A], mc[A], ml[A])
            minL = min(mr[A], br[A], bc[A])
            if minL > maxD:
                getLightest(mc, mr, br, bc)
            else:
                maxD = max(bc[A], mc[A], mr[A])
                minL = min(ml[A], tl[A], tc[A])
                if minL > maxD:
                    getLightest(mc, ml, tl, tc)

            return pixel

        self.changeEachPixel(self.dstImg, callBack)

    # getGradient compute the gradient of the image and store it to the Alpha channel
    def getGradient(self):
        B, G, R, A = 0, 1, 2, 3
        if self.fm == True:

            def callBack(i, j, pixel):
                if i == 0 or j == 0 or i == self.H - 1 or j == self.W - 1:
                    return pixel

                Grad = abs(
                    self.dstImg[i + 1, j - 1][A]
                    + 2 * self.dstImg[i + 1, j][A]
                    + self.dstImg[i + 1, j + 1][A]
                    - self.dstImg[i - 1, j - 1][A]
                    - 2 * self.dstImg[i - 1, j][A]
                    - self.dstImg[i - 1, j + 1][A]
                ) + abs(
                    self.dstImg[i - 1, j - 1][A]
                    + 2 * self.dstImg[i, j - 1][A]
                    + self.dstImg[i + 1, j - 1][A]
                    - self.dstImg[i - 1, j + 1][A]
                    - 2 * self.dstImg[i, j + 1][A]
                    - self.dstImg[i + 1, j + 1][A]
                )

                rst = self.unFloat(Grad / 2)
                pixel[A] = 255 - rst
                return pixel

        else:

            def callBack(i, j, pixel):
                if i == 0 or j == 0 or i == self.H - 1 or j == self.W - 1:
                    return pixel

                Grad = (
                    (
                        self.dstImg[i + 1, j - 1][A]
                        + 2 * self.dstImg[i + 1, j][A]
                        + self.dstImg[i + 1, j + 1][A]
                        - self.dstImg[i - 1, j - 1][A]
                        - 2 * self.dstImg[i - 1, j][A]
                        - self.dstImg[i - 1, j + 1][A]
                    )
                    ** 2
                    + (
                        self.dstImg[i - 1, j - 1][A]
                        + 2 * self.dstImg[i, j - 1][A]
                        + self.dstImg[i + 1, j - 1][A]
                        - self.dstImg[i - 1, j + 1][A]
                        - 2 * self.dstImg[i, j + 1][A]
                        - self.dstImg[i + 1, j + 1][A]
                    )
                    ** 2
                ) ** (0.5)

                rst = self.unFloat(Grad)
                pixel[A] = 255 - rst
                return pixel

        self.changeEachPixel(self.dstImg, callBack)

    # pushGradient will make the linework of the image sharper guided by the gradient in Alpha channel
    def pushGradient(self):
        B, G, R, A = 0, 1, 2, 3

        def getLightest(mc, a, b, c):
            mc[R] = mc[R] * (1 - self.sg) + (a[R] / 3 + b[R] / 3 + c[R] / 3) * self.sg
            mc[G] = mc[G] * (1 - self.sg) + (a[G] / 3 + b[G] / 3 + c[G] / 3) * self.sg
            mc[B] = mc[B] * (1 - self.sg) + (a[B] / 3 + b[B] / 3 + c[B] / 3) * self.sg
            mc[A] = 255
            return mc

        def callBack(i, j, pixel):
            iN, iP, jN, jP = -1, 1, -1, 1
            if i == 0:
                iN = 0
            elif i == self.H - 1:
                iP = 0
            if j == 0:
                jN = 0
            elif j == self.W - 1:
                jP = 0

            tl, tc, tr = (
                self.dstImg[i + iN, j + jN],
                self.dstImg[i + iN, j],
                self.dstImg[i + iN, j + jP],
            )
            ml, mc, mr = (
                self.dstImg[i, j + jN],
                pixel,
                self.dstImg[i, j + jP],
            )
            bl, bc, br = (
                self.dstImg[i + iP, j + jN],
                self.dstImg[i + iP, j],
                self.dstImg[i + iP, j + jP],
            )

            # top and bottom
            maxD = max(bl[A], bc[A], br[A])
            minL = min(tl[A], tc[A], tr[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, tl, tc, tr)

            maxD = max(tl[A], tc[A], tr[A])
            minL = min(bl[A], bc[A], br[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, bl, bc, br)

            # subdiagonal
            maxD = max(ml[A], mc[A], bc[A])
            minL = min(tc[A], tr[A], mr[A])
            if minL > maxD:
                return getLightest(mc, tc, tr, mr)
            maxD = max(tc[A], mc[A], mr[A])
            minL = min(ml[A], bl[A], bc[A])
            if minL > maxD:
                return getLightest(mc, ml, bl, bc)

            # left and right
            maxD = max(tl[A], ml[A], bl[A])
            minL = min(tr[A], mr[A], br[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, tr, mr, br)
            maxD = max(tr[A], mr[A], br[A])
            minL = min(tl[A], ml[A], bl[A])
            if minL > mc[A] and mc[A] > maxD:
                return getLightest(mc, tl, ml, bl)

            # diagonal
            maxD = max(tc[A], mc[A], ml[A])
            minL = min(mr[A], br[A], bc[A])
            if minL > maxD:
                return getLightest(mc, mr, br, bc)
            maxD = max(bc[A], mc[A], mr[A])
            minL = min(ml[A], tl[A], tc[A])
            if minL > maxD:
                return getLightest(mc, ml, tl, tc)

            pixel[A] = 255
            return pixel

        self.changeEachPixel(self.dstImg, callBack)

    # show the dstImg
    def show(self):
        cv2.imshow("dstImg", self.dstImg)
        cv2.waitKey()

    # changeEachPixel will traverse all the pixel of the image, and change it by callBack function, all the change will be applied after traversing
    def changeEachPixel(self, img, callBack):
        tmp = img.copy()
        for i in range(self.H):
            for j in range(self.W):
                tmp[i, j] = callBack(i, j, img[i, j].copy())
        self.dstImg = tmp

    # unFloat convert float to uint8,range from 0-255
    def unFloat(self, n):
        n += 0.5
        if n >= 255:
            return 255
        elif n <= 0:
            return 0
        return n

    # ShowInfo will show the basic infomation of the image
    def showInfo(self):
        print("Width: %d, Height: %d" % (self.orgImg.shape[1], self.orgImg.shape[0]))
        print("----------------------------------------------")
        print(
            "Input: %s\nPasses: %d\nFast Mode: %r\nStrength color: %g\nStrength gradient: %g"
            % (self.srcFile, self.ps, self.fm, self.sc, self.sg)
        )
        print("----------------------------------------------")

    # save image to disk
    def saveImage(self, filename="./output.png"):
        cv2.imwrite(filename, self.dstImg)

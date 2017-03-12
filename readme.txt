1 增加category
(1) 当前目录下创建一个category名字的目录名
(2) 在category目录下建一个index.md的文件，注意修改文件相应的category标识
(3) 修改_data/navigation.yml文件，增加对应category的映射

2 发布blog
(1) 如果对应的category没有建立，参考1建立
(2) 如果_posts下面没有对应的category，则创建一个目录
(3) 到对应的目录下创建博文，注意博文的命名格式: 日期-博文名.md. 例如: 2017-03-12-dev-test.md
(4) 博文中需要引用图片，放到images目录下
(5) 运行release.sh

# Nguyen_Huu_Vuot_19020489_SKLEARN
# Thuật toán Tf-IDf
## A) tiền xử lí dữ liệu bằng phương pháp biểu thị sự phân tán
* bài toán tính tần suất phân bố , xuất hiên của mỗi từ trên một chủ để (topic) hay một đoạn văn bản khác nhau
* tính toán tần suất dựa trên công thức tf-idf
* từ có tần xuất xuất hiện càng lớn thì sự quan trọng trong việc phân lớp càng cao
* các thông số tính
![alt](https://nguyenvanhieu.vn/wp-content/uploads/2019/01/tf.png)
* trong đó :
* f(t,d) - số lần xuất hiện của từ t trong văn bản d
* max{f(w,d):w thuộc d} - số lần xuất hiện nhiều nhất của một từ bất kì trong văn bản
**IDF** 
* tấn số nghịch của một từ trong văn bản
* tính IDF để giảm giá trị của những từ phổ biến (các phó từ, liên từ..) . Mỗi từ chỉ có 1 giá trị IDE duy nhất trong tập văn bản
![alt](https://nguyenvanhieu.vn/wp-content/uploads/2019/01/idf.png)
* trong đó:
* |D| : - tổng số văn bản trong tập D
* |{d thuộc D : t thuộc d}| - số văn bản chưa từ nhất định, với điều kiện t xuất hiện trong văn bản d, nếu từ đó ko xuất hiện thì mẫu sẽ gán mặc định bằng 1
* ==> khai quát nên công thức
![alt](https://wikimedia.org/api/rest_v1/media/math/render/svg/d1893056bff41c7829cf3023a5febda10f43e555)
*Những từ có giá trị TF-IDE cao là những từ xuất hiện nhiều trong văn bản này , xuất hiện ít trong các văn bản khác, những từ đó sẽ trở thành đặc trưng cho lớp văn bản hiện tại, từ đó xây dựng nên các vector số phục vụ cho phân lớp*

## Link tham khảo
* https://codetudau.com/machine-learning-nlp-scikit-learn/index.html
* https://codetudau.com/gioi-thieu-tien-xu-ly-trong-xu-ly-ngon-ngu-tu-nhien/
* https://codetudau.com/bag-of-words-tf-idf-xu-ly-ngon-ngu-tu-nhien/

# CFWithFeature
https://papers.nips.cc/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html 에 영감을 얻어 작성되었음 </br>
기존의 Matrix Factorizing 기법의 Collaborative Filtering은, User나 Item의 특징을 담을 수 없는 어려움이 있음. </br>

뉴럴 네트워크 방법론들 역시, Matrix Factorize를 딥러닝으로 하냐, 내적계산으로 하냐의 차이를 가질 뿐이지, 여전히 Feature 정보는 얹기 어려움. </br>

기존의 뉴럴 방법론을 사용하던, WMF를 사용하던, User와 Item을 행렬분해 할 수 있다고 가정했을때. 그 각각의 Matrix는 임베딩이라고 생각해볼 수 있음. </br>
그 임베딩에 각 Feature 정보를 Concat하여 적절히 비선형 회귀를 진행하고, WMF 된 R Matrix를 복원시키는 것(labels)으로, 회원, 아이템 임베딩, 그리고 각각의 Sparse Feature Vector들을 고려한 Collaborative Filtering을 진행할 수 있고, 새로운 아이템이나, 고객이 들어오더라도 강인한 예측을 제공할 수 있음. </br>

논문의 구현체에서는 Cold User, Cold Item 을 위한 Data Dropout을 진행하는데, 현재 구현체에서는 제공되지 않으며 아마 이 때문에 원본 복구에만 급급해서, 성능이 조금 떨어지는 경향이 있는 것 같습니다. 즉, 봤던거 위주로 예측해줍니다. </br>
저는 Neural CF를 사용해서, WMF를 진행하는 소스와 해당하는 값을 임베딩에 매핑하는 기능은 제공하지 않습니다. 다만, Neural CF 형태로 학습된 User Item 임베딩이 있다면, 그것을 load해서 사용하는 것으로 처리됩니다. </br>

Dropout을 적절히 진행해서, Perplexity를 높히는게 중요할 것 같은데, DropoutNet에서 Output인 내적값(WMF의 R_Matrix)을 구하는 과정을 단순 행렬곱으로 처리하지 않고, Denoising AutoEncoder를 사용하는 것으로, 데이터에 변화를 주지 않고도 유사한 성능을 내게 만들 수 있지 않을까 하는 아이디어 정도는 가지고 있습니다. </br>

구현은 DropoutNet의 구현체인 https://github.com/layer6ai-labs/DropoutNet 를 참고하였습니다.


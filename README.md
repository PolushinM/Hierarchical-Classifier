Тестовое задание, которое я сдавал на отборочный этап стажировки в Kazan Express.

### Описание задачи:
В маркетплейс каждый день поступает множество новых товаров и каждый из них необходимо отнести в определенную категорию в нашем дереве категорий. На это тратится много сил и времени, поэтому мы хотим научиться предсказывать категорию на основе названий и параметров товаров. \
Дано: около 3000 категорий, 290000 товаров в трейне и 70000 в тесте.

#### В полученном для работы [датасете](https://drive.google.com/drive/folders/194JOoKDZCkmpBglf7Fs7hlzk5xXJSYgI?usp=sharing) присутствуют следующие свойства объектов:
**id** - идентификатор товара: использую как идентификатор объекта в словарях, хранящих данные для обучения \
**title** - заголовок: скорее всего, несёт больше всего информации, поэтому его вес в документе, который используется для вычесления эмбеддингов, увеличен. \
**short_description** и **name_value_characteristics** - краткое описание и характеристики - объединяется в одну строку (Document) с title, затем, выполняется токенизация и лемматизация при помощи библиотек пакета natasha. \
**rating** и **feedback_quantity** - средний рейтинг товара и  количество отзывов по товару пока для предсказания не используются. Этому есть 2 причины: во-первых, интуиция подсказывает, что эти два свойства не имеют предсказательной силы в поставленной задаче, во-вторых, на момент добавления новой позиции в каталог, этой информации, как правило, ещё нет. Решил не тратить на это время (UPD: а потом оказалось, что это было "закладкой" организаторов - их нельзя было использовать).

Одна из особенностей - большой дисбаланс: есть множество пустых категорий, категорий, в которые попадают всего 2 товара, есть категории с тысячами товаров. \
Вторая важная особенность: тексты очень простые, не наполнены смыслом, раскрывающимся из контекста, это близко к набору ключевых слов, значит, простые модели на основе "мешка слов" должны работать не хуже сложных. \
Датасет очень большой, если данные перемешать, то можно использовать произвольный небольшой "кусок" для тестирования. Я взял для теста 4000 товаров из 290000, проверял метрики на разных подвыборках из указанного тест-сета, они отличались в третьем знаке после запятой. Можно даже не обучать модель на полном сете.

### Бейзлайн.
*[FastText_baseline.ipynb](https://github.com/PolushinM/Hierarchical-Classifier/blob/main/FastText_baseline.ipynb)* \
В качестве бейзлайна решено взять максимально простую модель - плоский классификатор на основе FastText. \
Таргет - листовая категория, путь в каталоге никак не учитывается при обучении и предсказании. \
Принцип работы FastText коротко можно описать так: Выделение n-чарграмм и n-вордграмм -> Получение (обучение с учителем) ембеддингов word2vec -> Усреднение полученных векторов слов по предложению (документу) -> Простой перцептрон (вернее, всего один слой с активацией Softmax). \
Важным улучшением стало увеличение веса первых слов описания title товара, как несущих самую важную информацию во всём описании.
После подбора гиперпараметров удалось достичь hF1=0.92 на тестовой выборке.

#### Что можно улучшить по сравнению с бейзлайном:
1. Биграммы могут быть не лучшим выбором способа извлечения фичей для векторизации. Нужно попробовать использовать прдвинутые инструменты, специализированные для русского языка. Можно попробовать лемматизацию и полдучение эмбеддингов слов средствами пакета natasha.
2. Так как слова, которые исполдьзуются в описании товара составляют относительно небольшой словарь, а взаимосвязь между словами в описаниях выражена менее ярко, чем в художественных и научных текстах, здесь может хорошо сработать "мешок слов", т.е. нужно попробовать кодировать документы при помощи тематической модели.
2. Никак не учитывается иерархия каталога. Необходимо попробовать построить алгоритм иерархической классификации, выполняющий обход дерева от корня к листьям, как это обычно делает человек. В этом случае, нужно научиться рассчитывать вероятности отнесения конкретного товара к каждому узлу дерева, для того чтобы алгоритм мог выбирать, в какой из дочерних узлов перейти.
3. Никак не учитывается наименование категории в каталоге. Можно разработать алгоритм получения вектора скрытых компонент, "эмбеддинга", из описания пути категории и использовать его для сравнения с эмбеддингом товара.
4. Простое усреднение по документу может работать плохо, так как самые важные слова находятся, обычно, в начале описания товара. Необходимо попробовать взвешенное усреднение.
5. Один из путей построения модели, предсказывающей вероятность, это: глобальный классификатор, получающий на вход эмбеддинг товара и эмбеддинг узла (эмбеддинг узла можно также считать, как среднее эмбеддингов товаров, попавших в него + можно добавить эмбеддинг текстового описания категории в каталоге). Простейший классификатор можно построить на сравнении косинусной близости эмбеддингов, более сложный - GBMT.
6. Второй путь - использование локального классификатора в каждом узле. Так как узлов много, это должна быть легковесная модель, такая как линейная регрессия. На вход такой модели подаётся только эмбеддинг товара.
7. Третий путь - попытаться объединить предыдущие два подхода методом стекинга или беггинга.

### Иерархический классификатор:
*[Hierarhical_no_catboost.ipynb](https://github.com/PolushinM/Hierarchical-Classifier/blob/main/Hierarchical_no_catboost.ipynb) и [Hierarhical_with_catboost.ipynb](https://github.com/PolushinM/Hierarchical-Classifier/blob/main/Hierarchical_with_catboost.ipynb)*

Основные модели, которые использовались при построении иерархического классификатора:

**LDA** (использовал gensim, пробовал BigARTM, но остановился на более простом варианте) для получения "тематических эмбеддингов" описания товара. \
Исходные данные для построения - список текстов-документов (Document), каждый из которых соответствует одному товару, и получен путём лемматизации и объединения в мешок слов из title (дважды), short_description и name_value_characteristics.

**Word2vec (GloVe) модель из библиотеки navec** из пакета natasha. Аналогично тематической модели, формируются лемматизированные тексты. Эмбеддинг документа рассчитывается как экспоненциально взвешенное среднее (с регулируемым параметром alpha) эмбеддингов слов в документе, первым словам даётся больший вес, последним - минимальный. navec выдаёт векторы размерности 300, для увеличения производительности и борьбы с переобучением в узлах, имеющих мало примеров, размерность снижена методом PCA (итоговая размерность настраивается как гиперпараметр).

Конкатенация полученных при помощи LDA и w2v (а потом добавил и эмбеддинги fasttext) эмбеддингов составляет эмбеддинг товара.

Каждому узлу дерева каталогов присваивается свой эмбеддинг, который равен среднему эмбеддингов, попавших в него товаров (вектор нормируется на собственную длину, чтобы сохранить только направление вектора - это необходимо в связи с влиянием регрессии к среднему при усреднении эмбеддингов, вследствие чего, среднее в узлах с большим числом товаров получается меньше по модулю. Кроме того к этому эмбеддингу добавляется, с некоторым весом, эмбеддинг, полученный из текстового описания пути категории в каталоге (например, "Одежда Женская Одежда Шапки Детские шапки").


#### Глобальный классификатор на основе GBMT:
*На вход* подаётся эмбеддинг документа и эмбеддинг узла. \
*Выход* - вероятность того, что товар принадлежит узлу. \
*Обучающая выборка:* \
в качестве положительных примеров - эмбеддинги узлов и товаров, входящих в данный узел, \
в качестве отрицательных примеров - эмбеддинги узла и товаров, входящих в соседние (siblings) узлы. \
Формируется один большой датасет (получилось почти 7 млн примеров), после чего, производится обучение одной модели для всех узлов и всех товаров.

#### Локальный классификатор на основе логистической регрессии:
Обучается отдельно для каждого узла (в узле, естественно, хранится только вектор весов модели и intercept) \
*На вход* подаётся эмбеддинг товара, мера косинусной близости эмбеддинга документа с эмбеддингом узла и логит предсказания глобального классификатора - в качестве дополнительных фичей. \
*Обучающая выборка:* \
в качестве положительных примеров - эмбеддинги товаров, входящих в данный узел + доп.фичи, \
в качестве отрицательных примеров - эмбеддинги товаров, входящих в соседние (siblings) узлы + доп.фичи. \
*Выход* - вероятность того, что товар принадлежит узлу (рассчитывается просто как сигма от скалярного произведения 2х векторов). \
Так как дисбаланс по количеству товаров в узлах очень велик (минимум - два товара, максимум - десятки тысяч), возникает проблема связанная с тем, что в узлах с малым количеством обучающих примеров (намного меньше количества фичей) требуется очень жесткая регуляризация, в узлах с большим количеством примеров - наоборот, регуляризация может сильно понизить качество. Поэтому параметр L2 регуляризации выбирается в зависимости от размера выборки, что регулируется двумя гиперпараметрами + выполняется небольшая кросс-валидация в каждом узле.

#### Алгоритм обхода дерева катологов:
Это алгоритм, который путём прохода по дереву от корня, определяет лист, в который попадает товар. \
Алгоритм "смотрит" на два уровня дерева вниз. \
На каждом шаге, подсчитывается вероятность попадания товара в дочерние узлы и в узлы, являющиеся дочерними дочерних (два раза одно и тоже не считается, значения кэшируются). \
Далее, производится переход в дочерний узел, получивший наивысшую оценку вероятности. \
Процесс повторяется, пока не дойдёт до листа.


### Построенная модель иерархического классификатора базируется на трёх классах:
**(для удобства объединены в библиотеку HierarchicalLibrary)**

***class TextProcessor***: класс, который управляет расчетами векторов скрытых представлений текстов, "эмбеддингов". \
Основные методы класса: \
*lemmatize_data* - принимает датафрейм с документами, делает лемматизацию \
*make_embeddibgs_dict* - создает словарь, где ключами являются id товаров, а значениями - эмбеддинги (уже объединённые LDA+Word2vec) документов (т.е. описаний товара) \
*get_embeddings* - принимает массив документов и возвращает массив эмбеддингов. \

***class CategoryTree***: класс, который хранит все узлы, необходимую информацию для обучения, а также реализует алгоритмы заполнения дерева, обхода при инференсе для определения категории товара. \
Дерево хранится в виде словаря, где ключами являются идентификаторы узлов. Так как в каждом узле хранятся id как родительского, так и дочерних узлов, это обеспечивает константную сложность перехода от одного узла к другому как при прямом, так и при обратном проходе по дереву. \
Основные методы класса: \
*add_nodes_from_df* - добавляет в дерево узлы из датафрейма, описывающего дерево каталога \
*add_goods_from_df* - добавляет в дерево товары из датафрейма - для обучения модели \
*update_embeddings* - добавляет в каждый узел эмбеддинг, равный среднему эмбеддингов товаров, попавших в узел \
*mix_in_description_embs* - "примешивает" к эмбеддингам узлов эмбеддинги, полученные энкодером из текстовых описаний узлов, с определённым весом \
*choose_leaf* - получает эмбеддинг товара, и выбирает для него наиболее подходящую листовую категорию \
*hF1_score* - рассчитывает иерархическую F1-меру \
*fit_local_weights* - обучает линейную регрессию в каждом узле

***class Classifier***: управляет процессом получения вероятностей принадлежности товара к узлу \
Основные методы: \
*fit_node_weights* - обучение одной линейной регрессии с учётом доп. фичей (глобальных классификаторов) \
*predict_local_proba* - предсказание вероятности нахождения товара в заданном узле по эмбеддингу товара \
*calc_global_train_array* - формирует массив данных для обучения глобального классификатора \
*fit_global_classifier* - обучает глобальный классификатор

Все три класса имеют множество методов для сохранения на диск и загрузки с диска обученных моделей и промежуточных результатов вычислений - это необходимо, так как вычисления могут выполняться достаточно долго.

**Энкодеры:**
Отдельно созданы 4 класса энкодеров, рассчитывающих эмбеддинги документов, совместимые с TextProcessor:
*LdaEncoder* - на основе LDA модели из библиотеки gensim
*NavecEncoder* - на основе предобученной GloVe модели navec с применением экспоненциального взвешивания
*FasttextEncoder* - на основе предобученной fasttext с применением экспоненциального взвешивания или без него (настраиваемый параметр)
*BertEncoder* - на основе предобученной модели BERT
Все энкодеры наследуются от базового абстрактного класса, который требует реализацию двух методов: load_model для загрузки модели с диска и transform для  расчета эмбеддингов списка документов.

### Полученные результаты:
- Метрика [иерархической F-меры](https://www.cs.kent.ac.uk/people/staff/aaf/pub_papers.dir/DMKD-J-2010-Silla.pdf) получилась ниже бейзлайна (максимум после всех экспериментов и доработок, указанных ниже, hF1=0,89), хотя модель значительно сложнее как вычислительно, так и архитектурно.
- Глобальный классификатор на основе GBMT (использовался CatBoost) не дал прироста качества при количестве деревьев до 300. При этом модель оставалась недообученной, а проиводительность работы такой модели была очень низкой. Увеличение количества деревьев нецелесообразно, поэтому GBMT в качестве глобального классификатора в последней версии не использовался (в коде пока, на всякий случай, оставил).
- Даже при размерности эмбеддинга равной 500, обобщающая способность модели оказалась недостаточной (это заметно по тому, что метрики, рассчитанные на тренировочной и тестовой выборке почти не отличаются, кроме того, по результатам экспериментов, увеличение размерности ведёт к росту качества, модель не переобучилась, она недообучается), это означает, что стоит попробовать более сложные модели (с большим числом параметров).
- Алгоритм обхода дерева, определяющий вероятности с учетом вероятностей дочерних узлов, не даёт кардинального улучшения результата (возможно, причина в том, что при наборе обучающих примеров, учитываются только соседние узлы, примеры из узлов уровнем ниже не берутся - это сделано для увеличения производительности). UPD: добавление в обучающую выборку отрицательных примеров, выбранных случайно, прибавило 0,01..0,02 к hF1.
- Снижение размерности векторов word2vec, как и уменьшение количества тем LDA снижает качество предсказания, но не сильно, это можно использовать для увеличения производительности.
- Так же пробовал делать ансамбль моделей *[FastTextBoost.ipynb](https://github.com/PolushinM/Hierarchical-Classifier/blob/main/FastTextBoost.ipynb)*: делать предсказания двумя моделями, иерархическим классификатором и FastText, а затем, анализируя предсказанные вероятности простой эвристикой, выбирать для каждого товара предсказание более уверенной модели. Второй вариант ансамбля: обучить иерархический классификатор на ошибочных и неуверенных предсказаниях FastText, а потом применить алгоритм выбора предсказания из двух алгоритмов.  - Оба варианта дают прирост целевой метрики hF1 всего около 0,003, от этой идеи пришлось отказаться.
- Продвинутые инструменты выделения токенов и лемматизации natasha в это йзадаче не дают желаемого прироста качества, можно использовать самый простой gensim.utils.simple_preprocess.


#### Не взлетает, какие ещё идеи?
1. Использовать BERT. Это SOTA модель обработки текстов, основным преимуществом которой является более полное "понимание" значения слов в предложении с учётом контекста, определение смысла предложений. Не известно, даст ли использование этой модели значимый прирост качества на плохо связанных текстах описаний товаров, представленных в датасете, но попробовать можно. UPD: Использовал модель ruBERT-tiny2 в качестве энкодера для иерархического классификатора, дообучать саму модель не стал: результаты ниже, чем у бейзлайна на FastText *[Hierarhical_BERT.ipynb](https://github.com/PolushinM/Hierarchical-Classifier/blob/main/Hierarhical_BERT.ipynb)*
2. Обучить глобальный иерархический multilabel классификатор с выходами на каждый узел (по одному выходу на каждый, а не только на листовые), нейросеть, естественно. Такой классификатор будет выдавать вероятности принадлежности товара для каждого узла, а уже готовый алгоритм обхода дерева, может непосредственно выполнить проход по каталогу и отнесение товара к категории. Можно использовать тот же FastText для расчета вероятностей. UPD: Пробовал fasttext multilabel one vs all, из-за сильного дисбаланса, он обучается предсказывать только наиболее крупные узлы, с ходу - не работает. Нужно придумывать очень сложную балансировку.
3. FastText выдаёт неожиданно хорошие результаты на и три- четыре- чарграммах, би- три- вордграммах, можно использовать похожие модели или сам FastText для получения эмбеддингов, и построить на этой основе что-то более сложное. UPD: пробовал встроить обученный fasttext в иерархический классификатор в качестве энкодера для получения эмбеддингов - получил увеличение hF1 на 0,01..0,02, но это всё ещё хуже, чем плоский классификатор на fasttext.
4. Использовать плоский fasttext для предварительного выбора листов-кондидатов, затем, проходить по дереву снизу вверх, считая вероятности принадлежности товара к узлам и их соседям готовым иерархическим классификатором, выбирать по савокупности предсказанных вероятностей лист из кандидатов. UPD: способ дал примерно то же качество, что и baseline (хотя и позволяет "подтянуть" качество до бейзлайна, если fasttext недообучен).
5. Использовать обученный fasttext как энкодер, эмбеддинги подать в простую NN на pytorch, применив Custom Loss, который будет учитывать не только простое попадание в категорию, но и расстояние между предсказанной и истинной категориями по дереву. Здесь можно попробовать изменить взвешивание эмбеддингов по предложению, а также попробовать подавать на вход 2-3 эмбеддинга с разными методами взвешивания. UPD: Написал custom loss для pytorch *[PyTorch_custom_loss.ipynb](https://github.com/PolushinM/Hierarchical-Classifier/blob/main/PyTorch_custom_loss.ipynb)* результаты такие же, как у бейзлайна.

#### Сводная таблица результатов.
| №   | Model | Classifier type | Output model | hF1 |
| --- | ----------- | ----------- | ----------- |:----:| 
| 1. |	Fasttext | Flat | Fasttext native | 0.918
| 2. |	Fasttext encoder + PyTorch custom loss | Flat + Hierarchical loss function | PyTorch | 0.916 | 
| 3. |	Fasttext encoder + PyTorch |  Flat | PyTorch | 0.914 |
| 4. |	Hierarchical model + fasttext, navec, LDA encoders |	Hierarchical, Classifier per node | Tree traversal algorithm | 0.9 | 
| 5. |	Hierarchical model + navec, LDA encoders + Catboost global classifier	 | Hierarchical, Classifier per node, global classifier | Tree traversal algorithm | 0.86 | 
| 6. |	Hierarchical model + navec, LDA encoders | Hierarchical, Classifier per node | Tree traversal algorithm | 0.86 | 
| 7. |	Hierarchical model + BERT encoder | Hierarchical, Classifier per node | Tree traversal algorithm | 0.83 | 


#### Итоги
Для сабмита подобрал гиперпараметры и обучил бейзлайн (файл SUBMIT.ipynb) на полном датасете. \
Результат - **1 место** в лидерборде среди 50 участников, отправивших готовые тестовые. \
Простейший baseline, сделанный за пару часов + несколько часов grid search позволили выйти в лидеры, а две недели экспериментов со иерархическим классификатором сложной структуры не увенчались никаким успехом, кроме полученного опыта о том, что простым задачам - простые решения.

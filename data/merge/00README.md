## �ϊ��v���O����
### timex2offset.rb

1�����P�ʂ� TIMEX3 �� EVENT �^�O�̏��̏������蓖�Ă�

CREATION_TIME �̂ݗ�O�Ƃ��ĕt�^���Ȃ�

$ for i in BCCWJ-TIMEX/*.xml; do ruby timex2offset.rb $i > tmp/`basename $i| cut -c1-5`-timex.txt ; done

### ud2offset.rb

1�����P�ʂ� �s�ԍ������蓖�Ă�

$ for i in UD_Japanese/*.conll; do ruby ud2offset.rb $i > tmp/`basename $i| cut -c1-5`-ud.txt ; done

$ for i in UD_Japanese/*.conll; do cp $i tmp/`basename $i| cut -c1-5`-ud.conll ; done

### merge.rb

$ for i in tmp/*.conll; do ruby merge.rb tmp/`basename $i -ud.conll`-timex.txt tmp/`basename $i .conll`.txt $i > merged/`basename $i`; done
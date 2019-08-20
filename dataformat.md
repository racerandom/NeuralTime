- current (sentence-level)
 
	[sentence] 1-to-n [entity\_masks] n-to-n [(sour\_entity, targ\_entity, relation)]
 
	=> [(sentence, sour\_entity\_mask, targ\_entity\_mask, relation)]


- document-level

	doc 1-to-n ([sentence\_mask], [entity\_mask]) n-to-n [(sour\_entity, targ\_entity, relation)]
	


package net.datasa.test3.Service;

import org.springframework.stereotype.Service;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import net.datasa.test3.domain.dto.BookDto;
import net.datasa.test3.domain.entity.BookEntity;
import net.datasa.test3.repository.BookRepository;

@Service
@Transactional
@RequiredArgsConstructor
public class BookService {

	private final BookRepository bookRep;

	public void bookAdd(BookDto bookDto) {
		BookEntity bookEntity = new BookEntity();
		bookEntity.setISBN(bookDto.getISBN());
		bookEntity.setTitle(bookDto.getTitle());
		bookEntity.setAuthor(bookDto.getAuthor());
		bookEntity.setPublisher(bookDto.getPublisher());
		bookEntity.setPublishDate(bookDto.getPublishDate());
		bookEntity.setPrice(bookDto.getPrice());
		bookEntity.setDiscountRate(bookDto.getDiscountRate());

		bookRep.save(bookEntity);
	}

	public BookDto bookGet(String ISBN) {
		BookDto bookDto = new BookDto();

		BookEntity bookEntity = bookRep.findById(ISBN).orElse(null);

		if (bookEntity == null)
			return null;

		bookDto.setISBN(bookEntity.getISBN());
		bookDto.setTitle(bookEntity.getTitle());
		bookDto.setAuthor(bookEntity.getAuthor());
		bookDto.setPublisher(bookEntity.getPublisher());
		bookDto.setPublishDate(bookEntity.getPublishDate());
		bookDto.setPrice(bookEntity.getPrice());
		bookDto.setDiscountRate(bookEntity.getDiscountRate());

		return bookDto;
	}
}

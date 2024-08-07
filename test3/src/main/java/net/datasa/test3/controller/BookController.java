package net.datasa.test3.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import lombok.extern.slf4j.Slf4j;
import net.datasa.test3.Service.BookService;
import net.datasa.test3.domain.dto.BookDto;

@Controller
@Slf4j
public class BookController {
	
	@Autowired
	BookService bookService;
	
	@PostMapping("add")
	public String addBook(@ModelAttribute BookDto bookDto) {
		
		bookService.bookAdd(bookDto);
		return "redirect:/addBook";
	}
	
	@GetMapping("find")
	public String findBook(@RequestParam(name ="ISBN") String ISBN, Model model) {
		BookDto bookDto = bookService.bookGet(ISBN);
		if(bookDto != null) {
			model.addAttribute("book", bookDto);
		}
		model.addAttribute("ISBN", ISBN);
		return "findBook";
	}
	
}

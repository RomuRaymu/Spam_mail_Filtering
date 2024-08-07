package net.datasa.test3.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

import lombok.extern.slf4j.Slf4j;

@Controller
@Slf4j
public class HomeController {
	@GetMapping({"", "/", "/home"})
	public String home() {
		return "home";
	}

	@GetMapping("addBook")
	public String addBook() {
		return "addBook";
	}

	@GetMapping("findBook")
	public String findBook() {
		return "findBook";
	}
	
	@GetMapping("return")
	public String returnHome() {
		return "home";
	}
	
}

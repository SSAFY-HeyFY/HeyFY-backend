package com.ssafy.ssashinsa.heyfy.home.controller;

import com.ssafy.ssashinsa.heyfy.home.docs.HomeDocs;
import com.ssafy.ssashinsa.heyfy.home.dto.HomeDto;
import com.ssafy.ssashinsa.heyfy.home.service.HomeService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@Tag(name = "Home", description = "홈 화면 API")
public class HomeController {

    private final HomeService homeService;

    @HomeDocs
    @GetMapping("/home")
    public ResponseEntity<HomeDto> home() {

        HomeDto homeDto = homeService.getHome();

        return ResponseEntity.ok(homeDto);
    }
}

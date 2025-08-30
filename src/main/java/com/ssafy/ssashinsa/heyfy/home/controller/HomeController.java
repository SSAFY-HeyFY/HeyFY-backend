package com.ssafy.ssashinsa.heyfy.home.controller;

import com.ssafy.ssashinsa.heyfy.home.docs.HomeDocs;
import com.ssafy.ssashinsa.heyfy.home.dto.HomeDto;
import com.ssafy.ssashinsa.heyfy.home.service.HomeService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequiredArgsConstructor
@Tag(name = "Home", description = "홈 화면 API")
public class HomeController {

    private final HomeService homeService;

    @HomeDocs
    @PostMapping("/home")
    public ResponseEntity<HomeDto> home() {
        log.info("홈 화면 조회 요청");
        HomeDto homeDto = homeService.getHome();

        return ResponseEntity.ok(homeDto);
    }
}

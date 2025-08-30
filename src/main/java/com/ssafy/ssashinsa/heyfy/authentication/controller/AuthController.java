package com.ssafy.ssashinsa.heyfy.authentication.controller;

import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthRefreshDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthSignInDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.AuthSignUpDocs;
import com.ssafy.ssashinsa.heyfy.authentication.docs.SidRefreshDocs;
import com.ssafy.ssashinsa.heyfy.authentication.dto.*;
import com.ssafy.ssashinsa.heyfy.authentication.service.AuthService;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@Slf4j
@Tag(name = "계정 관리", description = "계정 관리 API")
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @AuthSignInDocs
    @PostMapping("/signin")
    public ResponseEntity<SignInSuccessDto> signIn(@RequestBody SignInDto signInDto) {
        log.info("로그인 시도: {}", signInDto.getStudentId());
        return ResponseEntity.ok(authService.signIn(signInDto));
    }

    @AuthSignUpDocs
    @PostMapping("/signup")
    public ResponseEntity<SignUpSuccessDto> signUp(@Valid @RequestBody SignUpDto signUpDto){
        log.info("회원가입 시도: {}", signUpDto.getStudentId());
        return ResponseEntity.ok(authService.signUp(signUpDto));
    }

    @AuthRefreshDocs
    @PostMapping("/refresh")
    public ResponseEntity<TokenDto> refreshAccessToken(@RequestHeader("Authorization") String authorizationHeader, @RequestHeader("RefreshToken") String refreshToken) {
        log.info("액세스 토큰 재발급 시도");
        return ResponseEntity.ok(authService.refreshAccessToken(authorizationHeader, refreshToken));
    }

    @SidRefreshDocs
    @PostMapping("/sid/refresh")
    public ResponseEntity<SidDto> issueSid(@RequestBody SecondaryAuthRequestDto requestDto) {
        log.info("SID 재발급 시도");
        SidDto result = authService.issueSidWithPinFailureLogic(requestDto.getPinNumber());
        return ResponseEntity.ok(result);
    }
}
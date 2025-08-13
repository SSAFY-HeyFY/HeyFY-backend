package com.ssafy.ssashinsa.heyfy.authentication.controller;

import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInDto;
import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInSuccessDto;
import com.ssafy.ssashinsa.heyfy.authentication.dto.TokenDto;
import com.ssafy.ssashinsa.heyfy.authentication.service.AuthService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    @PostMapping("/signin")
    public ResponseEntity<SignInSuccessDto> signIn(@RequestBody SignInDto signInDto) {
        return ResponseEntity.ok(authService.signIn(signInDto));
    }

    @PostMapping("/refresh")
    public ResponseEntity<TokenDto> refreshAccessToken(@RequestHeader("Authorization") String authorizationHeader, @RequestHeader("RefreshToken") String refreshToken) {
        System.out.println("요청 받음");
        return ResponseEntity.ok(authService.refreshAccessToken(authorizationHeader, refreshToken));
    }
}
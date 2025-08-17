package com.ssafy.ssashinsa.heyfy.authentication.controller;

import com.ssafy.ssashinsa.heyfy.authentication.dto.*;
import com.ssafy.ssashinsa.heyfy.authentication.service.AuthService;
import com.ssafy.ssashinsa.heyfy.swagger.response.ApiRefresh;
import com.ssafy.ssashinsa.heyfy.swagger.response.ApiSignIn;
import com.ssafy.ssashinsa.heyfy.swagger.response.ApiSignUp;
import com.ssafy.ssashinsa.heyfy.swagger.response.ErrorsCommon;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
@ErrorsCommon
public class AuthController {

    private final AuthService authService;

    @ApiSignIn
    @PostMapping("/signin")
    public ResponseEntity<SignInSuccessDto> signIn(@RequestBody SignInDto signInDto) {
        System.out.println("로그인 요청");
        return ResponseEntity.ok(authService.signIn(signInDto));
    }

    @ApiSignUp
    @PostMapping("/signup")
    public ResponseEntity<SignUpSuccessDto> signUp(@Valid @RequestBody SignUpDto signUpDto){
        return ResponseEntity.ok(authService.signUp(signUpDto));
    }

    @ApiRefresh
    @PostMapping("/refresh")
    public ResponseEntity<TokenDto> refreshAccessToken(@RequestHeader("Authorization") String authorizationHeader, @RequestHeader("RefreshToken") String refreshToken) {
        return ResponseEntity.ok(authService.refreshAccessToken(authorizationHeader, refreshToken));
    }

}
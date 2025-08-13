package com.ssafy.ssashinsa.heyfy.authentication.service;

import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInDto;
import com.ssafy.ssashinsa.heyfy.authentication.dto.SignInSuccessDto;
import com.ssafy.ssashinsa.heyfy.authentication.jwt.JwtTokenProvider;
import com.ssafy.ssashinsa.heyfy.common.CustomException;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class AuthService {
    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider jwtTokenProvider;

    public SignInSuccessDto signIn(SignInDto signInDto) {
        try {
            UsernamePasswordAuthenticationToken authenticationToken =
                    new UsernamePasswordAuthenticationToken(signInDto.getUsername(), signInDto.getPassword());

            Authentication authentication = authenticationManager.authenticate(authenticationToken);
            String accessToken = jwtTokenProvider.createAccessToken(authentication);
            return new SignInSuccessDto(accessToken);
        } catch (BadCredentialsException e) {
            throw new CustomException(ErrorCode.LOGIN_FAILED);
        }
    }

}

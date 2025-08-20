package com.ssafy.ssashinsa.heyfy.authentication.service;

import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.authentication.exception.AuthErrorCode;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {
    private static final Logger log = LoggerFactory.getLogger(CustomUserDetailsService.class);
    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // 현재 db에서 데이터 가져오도록 설정
        Users user = userRepository.findByStudentId(username).orElseThrow(() -> new CustomException(AuthErrorCode.LOGIN_FAILED));

        log.debug("유저 데이터 가져옴 : "+user.getExternalId()); // 기존 userId는 바이너리 데이터로 저장되어 있어 로그에 찍으면 깨져서 나옴
        return new User(
                user.getStudentId(),
                user.getPassword(),
                Collections.emptyList()
        );
    }
}